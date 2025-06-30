# CadQuery Code Generator Model 

## Overview

This project creates a CadQuery code generator model by fine-tuning vision-language models (InternVL3) on the [GenCAD‑Code Dataset][GenCAD‑Code Dataset] (147K pairs of Images/CadQuery code). The solution is inspired by the original dataset paper which fine-tuned a 14B LLM with vision capabilities, but uses more efficient models with fewer parameters since LLaVA architectures are somewhat outdated.

## Project Structure

```text
mecagent-technical-test/
├── eval.py                  # Evaluation script
├── eval_ft.py               # Evaluation script for trained model
├── fine_tuning.py           # LoRA fine-tuning implementation
├── utils.py                 # Utility functions for image preprocessing and dataset handling
├── metrics/
│   ├── valid_syntax_rate.py # VSR metric implementation
│   └── best_iou.py          # IoU metric implementation
```
## Results Summary

| Model | VSR | Mean IoU | Notes |
|-------|-----|----------|-------|
| InternVL3-8B (Zero-shot) | 0.010 | 0.001 | Baseline |
| InternVL3-2B (Fine-tuned) | 0.930 | 0.341 | Training |
| InternVL3-8B (Fine-tuned) | 0.940 | 0.409 | Training |

## Solution Approach

### Baseline Model

The baseline evaluation uses **InternVL3** models in zero-shot mode:
- **InternVL3-8B-Instruct** (out-of-the-box): VSR = 0.010, Mean IoU = 0.001
- This suggests that, out of the box, InternVL3 has learned general language–vision patterns but lacks any specific grounding in CAD concepts or shape priors. Even  prompt‑engineering, such as providing detailed examples of primitive operations, enumerating available methods, or explicitly specifying numeric placeholders—yielded only marginal improvements. In short, without fine‑tuning on the GenCAD‑Code dataset, the vision‑language backbone does not possess the task‑specific knowledge or geometric intuition necessary to bridge from pixels to precise CadQuery scripts.

### Enhanced Model – Fine‑tuning Results 

Fine‑tuned on the full training dataset with evaluation on the validation set:

- **InternVL3‑2B‑Instruct + LoRA**  
  - VSR = 0.930  
  - Mean IoU = 0.341  

- **InternVL3‑8B‑Instruct + LoRA**  
  - VSR = 0.940  
  - Mean IoU = 0.409  
  
### Key Observations

- Training reached a plateau where cross-entropy loss no longer provided sufficient signal for IoU improvement
- VSR errors became negligible and could reach 100% with more training, but might hurt IoU
- **Identified bottleneck**: Need reinforcement learning using IoU computation for better training signal and example-level feedback

## Usage Instructions

### Environment Setup

1. Install uv (or use your preferred package manager)
2. Run uv sync
3. Run source .venv/bin/activate

### Evaluation

#### Using eval.py (Basic Evaluation)

```bash
python eval.py --model_path <model_path> --num_samples <number_of_samples>
```
Use this evaluation without LoRA, for base model.

#### Using eval_ft.py (Enhanced Evaluation)

The eval_ft.py script allows to evaluate trained model:

```bash
python eval_ft.py \
    --model_path "OpenGVLab/InternVL3‑8B‑Instruct" \
    --lora_path "./checkpoints_internvl3_finetune_cad_2B_new" \
    --num_samples 100 \
    --batch_size 4 \
    --output_json "predictions.json"
```

**Key parameters:**

- `--model_path`: Base model path
- `--lora_path`: Path to fine-tuned LoRA weights
- `--num_samples`: Number of samples to evaluate
- `--batch_size`: Batch size for inference
- `--output_json`: Output file for predictions

### Fine-tuning

#### Training a New Model
```bash
torchrun --nproc_per_node=1 fine_tuning.py \
    --model_name "OpenGVLab/InternVL3‑8B‑Instruct" \
    --output_dir "./checkpoints_new_model" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32
```
You have to use torchrun even with one GPU, as DDP is used to speed up training.
--nproc_per_node=1 allows you to specify the number of GPU.

**Key parameters:**

- `--model_name`: Base model to fine-tune  
- `--output_dir`: Directory to save checkpoints  
- `--num_epochs`: Number of training epochs  
- `--batch_size`: Training batch size  
- `--learning_rate`: Learning rate for optimization  
- `--lora_rank`: LoRA rank parameter  
- `--lora_alpha`: LoRA alpha parameter  


### Metrics

The evaluation uses two key metrics:

1. **Valid Syntax Rate (VSR)**: Measures the percentage of generated code that executes without syntax errors
2. **Intersection over Union (IoU)**: Measures geometric similarity between generated and ground truth 3D models



## Future Enhancements

With more time, the following improvements would be implemented:

1. **Reinforcement Learning**: While cross‐entropy training teaches the model to produce syntactically valid CadQuery code, it does not directly incentivize the precise geometric fidelity needed to reconstruct the target shapes—in essence, the model learns how to write code, but not which exact code yields the correct mesh. To bridge this gap, we will introduce a reinforcement‐learning phase in which the reward is the Intersection‐over‐Union (IoU) between the voxelized output of the generated script and the ground‑truth mesh. By combining a syntax bonus for error‑free execution with a continuous IoU signal in a PPO loop (and annealing out the CE term), the model will receive direct feedback on geometric accuracy, enabling it to surpass the current IoU plateau while retaining a high Valid Syntax Rate.
2. **Advanced Prompting or Agent**: To further guide the model toward precise parameter selection and correct API usage, we can augment each prompt with a brief “thinking trace” that spells out the numerical decisions and lists the available CadQuery functions. For instance, a chain‑of‑thought might read: “Step 1: set box width to image bounding‑box width → width = 20; Step 2: center at origin → x = 0, y = 0, z = 0; Step 3: fillet top edges with radius = 2.” Immediately below, we include a mini‑reference of in‑scope methods—.box(x, y, z), .workplane(plane), .fillet(radius), .cutThruAll(), etc.—so the model knows exactly which calls are legal. By explicitly anchoring each numeric value and highlighting every function, we reduce token ambiguity and help the agent generate code that not only compiles but also reconstructs the intended geometry with high fidelity.
3. **Agent Training**: By turning the model into an interactive agent that executes and verifies its CadQuery code not just during training but also at inference time, we can dramatically boost reliability and usability. At runtime, after the model emits each code block—or even the full script—it automatically runs the snippet, voxelizes the resulting mesh, and checks its IoU against the input image or a quick reference geometry. If the score falls below a user‑defined threshold, the agent can either trigger a lightweight re‑generation of the problematic section (e.g. tweaking a numeric parameter) or flag the issue for human review. This “generate → execute → verify → (optionally) regenerate” loop ensures that every shipped piece of code is pre‑validated for both syntax and geometric fidelity, turning our CAD‑code generator into a self‑correcting assistant that’s robust in production as well as in training.

[GenCAD‑Code Dataset]: https://huggingface.co/datasets/CADCODER/GenCAD-Code "GenCAD‑Code: 163k image–CadQuery script pairs for CAD code generation"
