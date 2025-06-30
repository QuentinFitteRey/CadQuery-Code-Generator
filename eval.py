
import torch
import numpy as np
import argparse
import json  

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel

from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from metrics.best_iou import get_iou_best
from utils import preprocess_pil_image, extract_python_code, get_dataset_splits



def evaluate_batch(test_dataset, model, tokenizer, batch_size=4, num_samples_to_evaluate=np.inf, output_json_path=None):
    gt_codes, pred_codes = {}, {}
    num_samples = min(len(test_dataset), num_samples_to_evaluate)
    test_subset = test_dataset.select(range(num_samples))
    print(f"Evaluating on {num_samples} samples from the test set")
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        batch_indices = range(i, min(i + batch_size, num_samples))
        batch_data = [test_subset[j] for j in batch_indices]
        batch_pixel_values_list, batch_num_patches_list, batch_gt_codes = [], [], {}
        
        for idx, example in zip(batch_indices, batch_data):
            pixel_values = preprocess_pil_image(example['image'], max_num=12)
            batch_pixel_values_list.append(pixel_values)
            batch_num_patches_list.append(pixel_values.size(0))
            sample_id = f"sample_{idx}"
            batch_gt_codes[sample_id] = example['cadquery']
            
        pixel_values_tensor = torch.cat(batch_pixel_values_list).to(torch.bfloat16).cuda()
        questions = ['<image>\nGenerate CadQuery Python code to create this 3D model. Only output the code. When generating CadQuery code, always define or choose a workplane as your datum, explicitly sketch each 2D profile with moves, lines, arcs and a closing command, turn each closed loop into a solid by extrusion, revolution or loft with clear parameters, combine solids in the correct order using union, cut or intersect operations (or boolean flags), assign the final result to a named variable (for example “solid”) and never rely on non‑standard display calls—always use only official CadQuery methods and ensure every profile is closed before creating a solid.'] * len(batch_data)
        
        try:
            responses = model.batch_chat(tokenizer, pixel_values_tensor,
                                         num_patches_list=batch_num_patches_list,
                                         questions=questions,
                                         generation_config=generation_config)
        except Exception as e:
            print(f"Error during model batch generation: {e}")
            responses = ["# Generation failed"] * len(batch_data)
        
        for (sample_id, gt_code), response in zip(batch_gt_codes.items(), responses):
            pred_codes[sample_id] = extract_python_code(response)
            gt_codes[sample_id] = gt_code

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
        print(f"\nSaving detailed predictions to {output_json_path}")
        output_data = {}
        for sample_id in gt_codes:
            output_data[sample_id] = {
                'predicted_code': pred_codes.get(sample_id, '# Prediction not found'),
                'ground_truth_code': gt_codes.get(sample_id, '# Ground truth not found')
            }
        
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully saved prediction results to {output_json_path}")


def main(args):
    if not torch.cuda.is_available():
        print("No CUDA devices found. This script requires GPUs to run otherwise it will be very slow.")
        return

    print(f"Loading base model from: {args.model_path}")
    base_model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if args.lora_path:
        print(f"Loading LoRA adapter from: {args.lora_path}")
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_path
        )
        print("Successfully loaded LoRA adapter.")
    else:
        print("No LoRA adapter path provided. Evaluating the base model.")
        model = base_model

    model = model.to('cuda')
    print("Model and tokenizer loaded successfully.")

    train_ds, test_ds = get_dataset_splits()
    if test_ds and model and tokenizer:
        evaluate_batch(
            train_ds, 
            model, 
            tokenizer, 
            batch_size=args.batch_size, 
            num_samples_to_evaluate=args.num_samples,
            output_json_path=args.output_json_path  
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-8B-Instruct",
                        help="Path to the base pretrained model from Hugging Face.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="DO NOT USE IT. Use eval_ft.py instead.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_samples", type=int, default=256,
                        help="Number of samples to evaluate from the test set.")
    parser.add_argument("--output_json_path", type=str, default="predictions.json",
                        help="Path to save the output JSON file with predictions.")
                        
    args = parser.parse_args()
    main(args)