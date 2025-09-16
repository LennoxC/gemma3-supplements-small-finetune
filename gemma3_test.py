#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator
from gemma3_finetune import test_dataset, collate_fn   # reuse dataset + collator
from tqdm import tqdm

# === Variables ===
save_dir = "finetunes"
experiment_name = os.getenv('TEST_RUN_NAME')
if experiment_name is None:
    raise ValueError("Environment variable TEST_RUN_NAME must be set!")

model_path = os.path.join(save_dir, experiment_name)

# === Accelerator ===
accelerator = Accelerator(mixed_precision="bf16")

# === Model & Processor ===
model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_path)
model = accelerator.prepare(model)
model.eval()

# === DataLoader ===
test_dataloader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
)
test_dataloader = accelerator.prepare(test_dataloader)

# === Loss Function ===
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# === Evaluation ===
test_loss, total_samples = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        outputs = model(**batch)
        batch_loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, batch["labels"])
        test_loss += batch_loss.item() * batch["input_ids"].size(0)
        total_samples += batch["input_ids"].size(0)

avg_test_loss = test_loss / total_samples
print(f"\n✅ Average Test Loss: {avg_test_loss:.4f}")

# === Generate Sample Outputs ===
print("\n=== Sample Generations ===")
for i in range(3):  # take 3 random samples
    example = test_dataset[i]
    inputs = processor(
        text=processor.apply_chat_template(example["messages"], add_generation_prompt=True, tokenize=False),
        images=[m["content"] for m in example["messages"] if m["role"] == "user" and m["content"][0]["type"] == "image"],
        return_tensors="pt",
        padding=True
    ).to(accelerator.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    decoded = processor.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n--- Example {i+1} ---")
    print("Prompt:", inputs["input_ids"].shape, "(truncated text)")  # could pretty print if needed
    print("Generated:", decoded)
