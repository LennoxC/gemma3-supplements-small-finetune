#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator
from tqdm import tqdm
from dotenv import load_dotenv
from Dataset import OCRVQADataset, process_vision_info   # same dataset class
from prompt_loader import PromptLoader
from PIL import Image
from peft import PeftModel


# === Load environment ===
load_dotenv()
data_dir = os.getenv("DATA_DIR")
save_dir = "finetunes"
experiment_name = os.getenv("TEST_RUN_NAME")
if experiment_name is None:
    raise ValueError("Environment variable TEST_RUN_NAME must be set!")
model_path = os.path.join(save_dir, experiment_name)

dataset_path = os.path.join(data_dir, "output_cleaned.jsonl")

# === Build dataset ===
dataset_obj = OCRVQADataset(dataset_path)

# same splits as trainer
total_size = len(dataset_obj)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
generator = torch.Generator().manual_seed(42)

_, _, test_dataset = random_split(
    dataset_obj, [train_size, val_size, test_size], generator=generator
)

# === Processor ===
processor = AutoProcessor.from_pretrained(model_path)

# === Collator ===
def collate_fn(examples):
    texts, images = [], []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    labels = batch["input_ids"].clone()
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100
    batch["labels"] = labels

    return batch

# === Accelerator ===
accelerator = Accelerator(mixed_precision="bf16")

# === Model ===
base_model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, model_path, torch_dtype=torch.bfloat16, device_map="auto")

# === DataLoader ===
test_dataloader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
)
model, test_dataloader = accelerator.prepare(model, test_dataloader)
model.eval()

# === Loss Function ===
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# === Evaluation ===
test_loss, total_samples = 0, 0

global_batches = 0
eval_limit = 20

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        outputs = model(**batch)
        batch_loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, batch["labels"])
        test_loss += batch_loss.item() * batch["input_ids"].size(0)
        total_samples += batch["input_ids"].size(0)

        global_batches += 1

        if global_batches >= eval_limit:
            break


avg_test_loss = test_loss / total_samples
print(f"Average Test Loss: {avg_test_loss:.4f}")

# === Generate some samples ===
print("\n=== Sample Generations ===")
for i in range(3):
    example = test_dataset[i]
    text = processor.apply_chat_template(example["messages"], add_generation_prompt=True, tokenize=False)
    image_inputs = process_vision_info(example["messages"])

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    ).to(accelerator.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    decoded = processor.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n--- Example {i+1} ---")
    print("Prompt:", text[:200], "..." if len(text) > 200 else "")
    print("Generated:", decoded)
