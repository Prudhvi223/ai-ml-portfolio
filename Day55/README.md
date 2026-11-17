# Day 55: Hands-on LoRA Fine-Tuning

## Overview

This project demonstrates hands-on Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) with Hugging Face.

A pretrained DistilGPT2 model was adapted using a custom instruction dataset.

## Topics Covered

### 1. Pretrained Language Models
- Understanding DistilGPT2.
- Loading pretrained models using Hugging Face Transformers.
- Understanding causal language modeling.

### 2. Custom Instruction Dataset
- Creating instruction, input and output examples.
- Converting Python data into a Hugging Face Dataset.
- Formatting instruction examples.

### 3. Tokenization and Preprocessing
- Converting text into token IDs.
- Padding and truncation.
- Creating attention masks and labels.
- Understanding loss masking with -100.

### 4. LoRA Configuration
- Understanding LoRA rank and alpha.
- Configuring target modules.
- Applying LoRA using Hugging Face PEFT.
- Checking trainable parameters.

### 5. Model Training
- Configuring Hugging Face Trainer.
- Training LoRA adapters using a custom dataset.
- Understanding training loss and metrics.

### 6. Model Evaluation and Saving
- Generating responses using the fine-tuned model.
- Comparing pretrained and fine-tuned responses.
- Saving the trained LoRA adapter.
- Reloading the adapter for future inference.

## Model Details

- Base model: DistilGPT2
- Framework: PyTorch
- Fine-tuning method: LoRA
- PEFT library: Hugging Face PEFT
- Custom training examples: 10
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.1

## Training Results

| Metric | Result |
|---|---:|
| Total parameters | 82,060,032 |
| Trainable parameters | 147,456 |
| Trainable percentage | 0.1797% |
| Final training loss | 3.1142 |

These results come from a small educational experiment and are not a measure of generalization or production performance.

## Hardware and Libraries

- NVIDIA GeForce GTX 1650 (4 GB VRAM)
- Python
- PyTorch
- Transformers
- Datasets
- PEFT
- Accelerate
- TRL
- Jupyter Notebook

## Project Files

- `Hands_On_LoRA.ipynb`: Complete implementation and experiments.
- `distilgpt2-lora-adapter/`: Saved LoRA adapter and configuration.

## Key Learning Outcomes

- Understood the complete LoRA fine-tuning workflow.
- Created a custom instruction dataset.
- Applied LoRA to a pretrained language model.
- Trained adapter parameters without updating the original model weights.
- Saved and reloaded the trained adapter.

## Conclusion

Successfully completed a hands-on LoRA fine-tuning experiment using Hugging Face Transformers and PEFT. This project demonstrates how pretrained language models can be adapted to specific tasks with a small number of trainable parameters.
