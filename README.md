# Gemma 3 Supplements Finetune

This repo contains the code to finetune the `google/gemma-3-4b-it` model for food & beverage label OCR-VQA. Details on the project motivation and plan can be found in the [design report](https://github.com/LennoxC/Label-Verification-Reports/blob/main/Design%20Report/pdf/Automatic%20Label%20Verification%20Design%20Report.pdf).

## Environment
The data and model checkpoints are on a GPU server with 3x RTX A6000 NVIDIA GPUs (48gb VRAM per GPU, 32 CPU cores, 192gb ram). Two GPUs on this server are sufficient to train with a batch size of 2.

## Portability
This repo isn't designed to be a complete self-contained environment for fine-tuning. Setup is required to correctly configure:
- the dataset
- GPU settings, drivers & accelerate
- access to the Hugging Face API

## Results
Results are not yet available, but will be accessible in the [Label Verification Reports](https://github.com/LennoxC/Label-Verification-Reports) repo when complete.

## 
This is a project for AIML339 at Victoria University of Wellington.