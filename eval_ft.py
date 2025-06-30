import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from peft import PeftModel
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
from utils import preprocess_pil_image, extract_python_code, get_dataset_splits


@torch.no_grad()
def evaluate_batch(dataset, model, tokenizer, batch_size=4, num_samples_to_evaluate=np.inf, output_json_path=None):
    gt_codes, pred_codes = {}, {}
    num_samples = min(len(dataset), num_samples_to_evaluate)
    subset = dataset.select(range(num_samples))
    print(f"Evaluating on {num_samples} samples")

    generation_config = GenerationConfig(
        max_new_tokens=4096,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        batch_indices = range(i, min(i + batch_size, num_samples))
        batch_examples = [subset[j] for j in batch_indices]
        current_batch_size = len(batch_examples)

        batch_pixel_values_list, batch_num_patches, batch_prompts, batch_ids = [], [], [], []
        for idx, example in zip(batch_indices, batch_examples):
            pixel_values = preprocess_pil_image(example['image'], max_num=12)
            num_patches = pixel_values.size(0)
            
            question = '<image>\nGenerate CadQuery Python code to create this 3D model. Only output the code.'
            conv = model.conv_template.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_placeholder = conv.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            final_prompt = prompt_placeholder.replace('<image>', image_tokens, 1)

            batch_pixel_values_list.append(pixel_values)
            batch_num_patches.append(num_patches)
            batch_prompts.append(final_prompt)
            sample_id = f"sample_{idx}"
            batch_ids.append(sample_id)
            gt_codes[sample_id] = example['cadquery']

        pixel_values_tensor = torch.cat(batch_pixel_values_list).to(torch.bfloat16).to(model.device)
        batched_vit_embeds = model.extract_feature(pixel_values_tensor)

        tokenizer.padding_side = "left" 
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        input_embeds = model.language_model.get_input_embeddings()(input_ids)

        embed_idx_tracker = 0
        for i in range(current_batch_size):
            num_patches_for_item = batch_num_patches[i]
            vit_embeds_for_item = batched_vit_embeds[embed_idx_tracker : embed_idx_tracker + num_patches_for_item]
            
            placeholder_indices = (input_ids[i] == img_context_token_id)
            if placeholder_indices.any():
                input_embeds[i, placeholder_indices] = vit_embeds_for_item.reshape(-1, vit_embeds_for_item.shape[-1]).to(input_embeds.device)
            
            embed_idx_tracker += num_patches_for_item
            
        try:
            output_ids = model.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)


            for sample_id, response in zip(batch_ids, responses):
                pred_codes[sample_id] = extract_python_code(response)
        except Exception as e:
            print(f"Error during batched model generation: {e}")
            for sample_id in batch_ids:
                pred_codes[sample_id] = f"# Generation failed: {e}"

    print("Calculating Valid Syntax Rate (VSR)")
    vsr = evaluate_syntax_rate_simple(pred_codes)
    print(f"Valid Syntax Rate: {vsr:.3f}")

    print("Calculating Best IOU (this may take a while)")
    iou_scores = []
    for sample_id in tqdm(pred_codes.keys(), desc="Calculating IOU"):
        try:
            from metrics.valid_syntax_rate import _load_solid_from_code
            _load_solid_from_code(pred_codes[sample_id], sample_id)
            iou = get_iou_best(gt_codes[sample_id], pred_codes[sample_id])
            iou_scores.append(iou)
        except Exception:
            iou_scores.append(0.0)

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    print(f"Mean Best IOU: {mean_iou:.3f}")
    print(f"\n--- Evaluation Complete ---\nSummary: VSR = {vsr:.3f}, Mean IOU = {mean_iou:.3f}")

    if output_json_path:
        print(f"\nSaving predictions to {output_json_path}")
        output_data = {sid: {'predicted_code': pred_codes.get(sid), 'ground_truth_code': gt_codes.get(sid)} for sid in gt_codes}
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully saved prediction results to {output_json_path}")

def main(args):
    if not torch.cuda.is_available():
        print("No CUDA devices found. This script requires GPUs to run otherwise it will be very slow.")
        return

    print(f"Loading base model from: {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, 'conv_template'):
        tokenizer.chat_template = model.conv_template.render_jinja_template() if hasattr(model.conv_template, 'render_jinja_template') else str(model.conv_template)
        tokenizer.conv_template = model.conv_template

    if args.lora_path:
        print(f"Loading and applying LoRA adapter from: {args.lora_path}")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.lora_path,
            torch_dtype=torch.bfloat16,
            attn_implementation = "flash_attention_2" #NEED FLASH ATTENTION 2 FOR THIS , please comment if not
        )
        print("Successfully applied LoRA adapter to the language model.")
    else:
        print("No LoRA adapter path provided. Evaluating the base model.")

    model = model.to('cuda')
    print("Model and tokenizer loaded successfully.")

    train_ds, test_ds = get_dataset_splits()
    if test_ds:
        evaluate_batch(
            test_ds, 
            model,
            tokenizer,
            batch_size=args.batch_size,
            num_samples_to_evaluate=args.num_samples,
            output_json_path=args.output_json_path
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-8B-Instruct", 
                        help="Path to the base pretrained model.")
    parser.add_argument("--lora_path", type=str, default="/home/hice1/qfitterey3/scratch/mecagent/checkpoints/checkpoints_internvl3_finetune_cad_8B/checkpoint-step-2700",
                         help="Path to the trained LoRA adapter checkpoint directory.")    
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for evaluation. Adjust based on VRAM.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate.")
    parser.add_argument("--output_json_path", type=str, default="predictions_optimized_8B_v3.json", help="Path to save the output JSON file.")

    args = parser.parse_args()
    main(args)