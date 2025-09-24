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
from utils import visualize_masking

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

    start_tokens = [105, 4368]  # <start_of_turn> followed by model token
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )

    for example in examples:
        messages = example["messages"]
        # Full chat template with all special tokens
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )
        text += processor.tokenizer.eos_token
        texts.append(text)

        # Image
        image_inputs = process_vision_info(messages)
        images.append(image_inputs)

    # Let processor tokenize + pad + add specials
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Clone input_ids as labels
    labels = batch["input_ids"].clone()

    # Mask everything except the model response
    for i in range(labels.size(0)):
        seq = labels[i]
        found = False
        for j in range(len(seq) - 1):
            if seq[j].item() == start_tokens[0] and seq[j+1].item() == start_tokens[1]:
                start_idx = j + 2  # start after <start_of_turn> + model token
                labels[i, :start_idx] = -100
                found = True
                break
        if not found:
            # mask entire sequence if no response found
            labels[i, :] = -100

        # Mask padding + image tokens
        labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100
        labels[i, labels[i] == image_token_id] = -100

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
    lora_alpha=8,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    task_type="CAUSAL_LM"
)

# Gemma 3 modules:
#       "q_proj",
#       "k_proj",
#       "v_proj",
#       "o_proj",
#       "gate_proj",
#       "up_proj",
#       "down_proj"

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# ================ Training Loop ================
# Params:
max_steps = 2001
num_warmup_steps = int(0.02 * max_steps)
gradient_accumulation_steps = 8
log_every = 2  # steps for scalar logging
val_every = 50  # run validation every N steps
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
best_val_loss = float("inf")
steps_since_improvement = 0

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
                val_loss = 0.0

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        outputs = model(**val_batch)
                        batch_loss = outputs.loss
                        val_loss += batch_loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                writer.add_scalar("eval/loss", avg_val_loss, global_step)

                # --- Track best model ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    steps_since_improvement = 0

                    # save best model
                    accelerator.wait_for_everyone()
                    output_dir = f"{save_dir}/{run_name}/best"
                    with deepspeed.zero.GatheredParameters((p for n, p in model.named_parameters() if "lora" in n)):
                        if accelerator.is_main_process:
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            processor.save_pretrained(output_dir)
                else:
                    steps_since_improvement += val_every

                # --- Early stopping ---
                if steps_since_improvement >= 500:
                    print(f"Early stopping at step {global_step}. Best val loss: {best_val_loss:.4f}")
                    break

                model.train()  # switch back

            # ---- Regular checkpointing every 200 steps ----
            if global_step % 200 == 0:
                accelerator.wait_for_everyone()
                ckpt_dir = f"{save_dir}/{run_name}/checkpoint-{global_step}"
                with deepspeed.zero.GatheredParameters((p for n, p in model.named_parameters() if "lora" in n)):
                    if accelerator.is_main_process:
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        processor.save_pretrained(ckpt_dir)

            if global_step >= max_steps:
                break

    # Early stopping outer break
    if steps_since_improvement >= 500 or global_step >= max_steps:
        break

# ============ SAVE FINAL MODEL ============
accelerator.wait_for_everyone()
final_dir = f"{save_dir}/{run_name}/final"
with deepspeed.zero.GatheredParameters((p for n, p in model.named_parameters() if "lora" in n)):
    if accelerator.is_main_process:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)

writer.close()
