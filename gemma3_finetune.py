#!/usr/bin/env python
# coding: utf-8

#Gemma3-4b fine-tune for supplements label OCR-VQA

import os
import json, random
import torch
from torch.utils.data import random_split
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from peft import LoraConfig
from dotenv import load_dotenv
from prompt_loader import PromptLoader
import numpy as np
import deepspeed

# ==================== ENVIRONMENT SETUP ====================

Image.MAX_IMAGE_PIXELS = None # disable decompression bomb warning on PIL images

load_dotenv() # load environmental variables

prompts = PromptLoader() # create the prompts loader class. This loads the prompts from the /prompts directory. These are tracked with git.

# === environmental variables ===

hf_token = os.getenv('HF_TOKEN') # load the hugging face token. This must have read/write access.
hf_home  = os.getenv('HF_HOME') # home directory for model checkpoints
data_dir = os.getenv('DATA_DIR') # where to load the images/jsonl file from
checkpoint_dir = os.getenv('CHECKPOINT_DIR') # checkpoints are saved here. Use a location with plenty of storage
logs_dir = os.getenv('LOGS_DIR') # logs are outputted here. Store near the checkpoints for tidyness
run_name = os.getenv('RUN_NAME') # name of this experiment

# check all the environmental variables are set
env_vars = {
    "HF_TOKEN": hf_token,
    "HF_HOME": hf_home,
    "DATA_DIR": data_dir,
    "CHECKPOINT_DIR": checkpoint_dir,
    "LOGS_DIR": logs_dir,
    "RUN_NAME": run_name
}
missing = [name for name, value in env_vars.items() if value is None]

if missing:
    for name in missing:
        print(f"{name} is null")
    exit("Environmental Variables " + ", ".join(env_vars.keys()) + " must be set.")

# === Variables for the project ===

dataset_path = os.path.join(data_dir, "output_cleaned.jsonl")
images_path  = os.path.join(data_dir, "images")
logs_dir = os.path.join(logs_dir, run_name)

base_model = "google/gemma-3-4b-it"
save_dir = "finetunes"

system_message = prompts.get_prompt("system")
user_prompt = prompts.get_prompt("user")

# ============== Datasets and Data Loaders ==============
from Dataset import OCRVQADataset, process_vision_info

dataset_obj = OCRVQADataset(dataset_path)

# === Set split sizes ===
total_size = len(dataset_obj)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset_obj, [train_size, val_size, test_size], generator=generator
)

# ============== MODELS AND TRAINERS ==============

accelerator = Accelerator(mixed_precision="bf16")
writer = SummaryWriter(log_dir=logs_dir)

model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

model.config.use_cache = False

processor = AutoProcessor.from_pretrained(base_model)

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts, images = [], []

    for example in examples:
        # Process images
        image_inputs = process_vision_info(example["messages"])
        images.append(image_inputs)

        # Get full chat text
        text = processor.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
            do_pan_and_scan=True
        ).strip()
        texts.append(text)

    # Encode the batch
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    labels = batch["input_ids"].clone()

    # IDs for the sequence indicating start of model response
    start_tokens = [105, 4368]  # <start_of_turn>, model
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )

    for i in range(labels.size(0)):
        seq = labels[i]

        # loss should only be computed over the JSON model output
        # to do this, we mask all the tokens up to <start_of_turn>model
        # <start_of_turn>model is followed by the model response
        # <start_of_turn>model is two tokens
        # Find first occurrence of the consecutive tokens
        found = False
        for j in range(len(seq) - 1):
            if seq[j].item() == start_tokens[0] and seq[j+1].item() == start_tokens[1]:
                start_idx = j + 1
                labels[i, :start_idx+1] = -100
                found = True
                break
        if not found:
            # if model response is not in training data, mask all tokens (nothing to compute loss on)
            labels[i, :] = -100

    # Mask padding and image tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100  # optional extra masking

    batch["labels"] = labels
    return batch

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# ================ Training Loop ================
# Params:
max_steps = 1001
num_warmup_steps = int(0.05 * max_steps)
gradient_accumulation_steps = 8
log_every = 2  # steps for scalar logging
val_every = 25  # run validation every N steps
max_grad_norm = 1.0

optimizer = torch.optim.AdamW(model.parameters())

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_steps,
)

model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

# === Start Training ===

model.train()

global_step = 0
while global_step < max_steps:
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps

        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # ---- Training Scalars ----
            if global_step % log_every == 0:
                writer.add_scalar("train/loss", loss.item() * gradient_accumulation_steps, global_step)

            # ---- Validation Mid-Epoch ----
            if global_step % val_every == 0:
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        outputs = model(**val_batch)
                        batch_loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, val_batch["labels"])
                        val_loss += batch_loss.item()

                avg_val_loss = val_loss / len(val_dataloader)

                writer.add_scalar("eval/loss", avg_val_loss, global_step)

                model.train()  # switch back

            if global_step >= max_steps:
                break


# ============ SAVE THE MODEL ============
accelerator.wait_for_everyone()

output_dir = f"{save_dir}/{run_name}"

with deepspeed.zero.GatheredParameters((p for n, p in model.named_parameters() if "lora" in n)): # required to prevent empty LoRA layers being saved
    if accelerator.is_main_process:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)


writer.close()

