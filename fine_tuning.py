import os
import argparse
import torchvision.transforms as T
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False




class CorrectedDataCollator:
    def __init__(self, tokenizer, transform, max_seq_length, num_image_token):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.num_image_token = num_image_token

    def __call__(self, raw_features):
        batch_input_ids, batch_labels, batch_pixel_values = [], [], []
        for example in raw_features:
            image = example['image']
            pixel_values = self.transform(image)
            batch_pixel_values.append(pixel_values)
            image_placeholder = f"<image>{'<__img_context__>' * self.num_image_token}</image>"
            prompt_text = "Generate the CadQuery Python code for this 3D model."
            user_prompt = f"{image_placeholder}\n{prompt_text}"
            model_response = example['cadquery']
            conv = self.tokenizer.conv_template.copy()
            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], model_response)
            full_prompt = conv.get_prompt() + self.tokenizer.eos_token
            assistant_role_marker = conv.roles[1]
            prompt_to_mask = full_prompt.split(assistant_role_marker)[0] + assistant_role_marker
            tokenized_full = self.tokenizer(full_prompt, max_length=self.max_seq_length, truncation=True, padding=False)
            tokenized_prompt = self.tokenizer(prompt_to_mask, max_length=self.max_seq_length, truncation=True, padding=False, add_special_tokens=False)
            input_ids = tokenized_full['input_ids']
            labels = list(input_ids)
            labels[:len(tokenized_prompt['input_ids'])] = [-100] * len(tokenized_prompt['input_ids'])
            if labels[-1] == self.tokenizer.eos_token_id:
                labels[-1] = -100
            batch_input_ids.append(torch.tensor(input_ids))
            batch_labels.append(torch.tensor(labels))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
        pixel_values_batch = torch.stack(batch_pixel_values).to(torch.bfloat16)
        return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": labels_padded, "pixel_values": pixel_values_batch}

def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    use_wandb = has_wandb and not args.no_wandb
    if rank == 0 and use_wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=args)

    if rank == 0: print(f"Loading model: {args.model_path}")
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, config=model_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    
    special_tokens_to_add = ['<image>', '</image>', '<__img_context__>', '<|im_start|>', '<|im_end|>']
    if tokenizer.add_tokens(special_tokens_to_add, special_tokens=True) > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.language_model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = 'left'
    tokenizer.conv_template = model.conv_template
    
    try:
        model.img_context_token_id = tokenizer.convert_tokens_to_ids('<__img_context__>')
        num_image_token = model.num_image_token
        if rank == 0: print(f"Successfully found model.num_image_token: {num_image_token}")
    except (AttributeError, KeyError):
        raise RuntimeError("Could not set up image tokens from the model/tokenizer.")

    if rank == 0: print("Applying LoRA to the `language_model` submodule...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules,
    )
    model.language_model = get_peft_model(model.language_model, peft_config)
    if rank == 0: model.language_model.print_trainable_parameters()
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((args.image_input_size, args.image_input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    data_collator = CorrectedDataCollator(tokenizer, transform, args.max_seq_length, num_image_token)
    
    if rank == 0: print("Loading raw dataset...")
    full_dataset = load_dataset("CADCODER/GenCAD-Code")
    train_dataset = full_dataset['train']
    val_dataset = full_dataset['validation']
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, collate_fn=data_collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.validation_batch_size, sampler=val_sampler, num_workers=args.num_workers, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                vit_embeds = model.module.extract_feature(batch['pixel_values'].to(device))
            
            input_embeds = model.module.language_model.get_input_embeddings()(input_ids)
            selected = (input_ids == model.module.img_context_token_id)
            if selected.sum() == 0: continue
            input_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
            
            outputs = model.module.language_model(
                inputs_embeds=input_embeds,
                attention_mask=batch['attention_mask'].to(device),
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(args.gradient_accumulation_steps)

                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log_loss = loss.item() * args.gradient_accumulation_steps
                    progress_bar.set_postfix({"loss": log_loss, "lr": current_lr})
                    if use_wandb:
                        wandb.log({
                            "train/loss": log_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch + ((step + 1) / len(train_dataloader))
                        }, step=global_step)

                if global_step > 0 and global_step % args.validation_steps == 0:
                    if rank == 0: print(f"\n--- Running validation at global step {global_step} ---")
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            vit_embeds = model.module.extract_feature(val_batch['pixel_values'].to(device))
                            input_ids = val_batch['input_ids'].to(device)
                            input_embeds = model.module.language_model.get_input_embeddings()(input_ids)
                            selected = (input_ids == model.module.img_context_token_id)
                            if selected.sum() == 0: continue
                            input_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
                            val_outputs = model.module.language_model(
                                inputs_embeds=input_embeds,
                                attention_mask=val_batch['attention_mask'].to(device),
                                labels=val_batch['labels'].to(device)
                            )
                            total_val_loss += val_outputs.loss.item()
                    
                    avg_val_loss = total_val_loss / len(val_dataloader)
                    loss_tensor = torch.tensor(avg_val_loss, device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    final_avg_loss = loss_tensor.item()
                    
                    if rank == 0:
                        print(f"--- Validation Complete ---")
                        print(f"Average Validation Loss: {final_avg_loss:.4f}")
                        if use_wandb:
                            wandb.log({"validation/loss": final_avg_loss}, step=global_step)
                        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        model.module.language_model.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)
                        print(f"Checkpoint saved to {checkpoint_path}")
                    
                    dist.barrier()
                    model.train()
    
    progress_bar.close()
    if rank == 0:
        print("Training complete. Saving final model.")
        if use_wandb: wandb.finish()
        final_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        model.module.language_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Classic DDP Fine-tuning for InternVL")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-2B-Instruct", help="Path to model.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/InternVL3-2B-Instruct_v1/", help="Output directory.")
    parser.add_argument("--image_input_size", type=int, default=448, help="Image input size.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length.")
    
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps to accumulate gradients. Batch size = batch_size * num_gpus * grad_accum_steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=252, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], help="Target modules for LoRA.")
    
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation every N optimizer steps.")
    parser.add_argument("--validation_batch_size", type=int, default=2, help="Per-GPU validation batch size.")
    parser.add_argument("--max_val_samples", type=int, default=1000, help="Max validation samples.")
    
    parser.add_argument("--project_name", type=str, default="internvl-cad-finetune", help="W&B project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Name of run in W&B.")
    parser.add_argument("--no_wandb", action='store_true', default=True, help="Disable W&B logging.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()