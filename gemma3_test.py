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
import matplotlib.pyplot as plt


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
dataset_obj = OCRVQADataset(dataset_path, train=False)

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

        print(text)  # optional: for debugging

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
eval_limit = 1

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

print("\n=== Example Predictions from Test Set ===")

# Pick a handful of samples
num_examples = 5
subset = [test_dataset[i] for i in range(num_examples)]

gen_model = accelerator.unwrap_model(model)
eos_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")

for i, example in enumerate(subset):
    # Build prompt
    image_inputs = process_vision_info(example["messages"])
    text_input = processor.apply_chat_template(
        example["messages"], add_generation_prompt=True, tokenize=False
    )

    # get the image name
    image_file_name = test_dataset.dataset.get_image_id(test_dataset.indices[i])

    inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt"
    ).to(accelerator.device)

    # Generate
    with torch.no_grad():
        generated_ids = gen_model.generate(
            **inputs,
            max_new_tokens=128,
            eos_token_id=eos_token_id
        )

    imgs = process_vision_info(test_dataset[i]["messages"])

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract only the model’s part
    #if "<start_of_turn>model" in decoded:
    #    generated_text = decoded.split("<start_of_turn>model")[-1].strip()
    #else:
    #    generated_text = decoded.strip()

    print(f"\n--- Example {i+1} ---")
    print(f"Image: {image_file_name}")
    print(f"Model Output:\n{decoded}")