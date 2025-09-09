#!/usr/bin/env python
# coding: utf-8

# # Gemma3-4b fine-tune for supplements label OCR-VQA

# In[1]:


#%pip install torch tensorboard
#%pip install transformers datasets accelerate evaluate trl protobuf sentencepiece
#%pip install flash-attn
#%pip install accelerate peft


# In[2]:


#%pip install tf-keras


# In[3]:


import os


# In[4]:


hf_token = os.getenv('HF_TOKEN')


# In[5]:


#%env HF_HOME=/local/scratch/crowelenn-aiml339/


# In[6]:


base_model = "google/gemma-3-4b-it"
checkpoint_dir = "/local/scratch/crowelenn-aiml339/checkpoints"
learning_rate = 5e-5


# Tensorboard setup

# In[7]:


#%pip install tensorboard


# In[8]:


system_message = "You are a quality control robot responsible for monitoring the quality of supplement labels."
user_prompt = """Using primarily the text contained in the attached label supplement image, answer the list of questions in the <QUESTIONS> tags.
Answer concisely in a JSON format with no preamble, allowing the response to easily be parsed. An example response would be:
{
  "brand": "label supplelments co",
  "contents": 120
}

<QUESTIONS>
"""


# In[9]:


def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample["questions"]
                        #"text": user_prompt.format(
                        #    questions=sample["questions"]
                        #),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answers"]}],
            },
        ],
    }


# In[17]:


import json, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OCRVQADataset(Dataset):
    def __init__(self, jsonl_file, transform=None, min_q=1, max_q=4):
        with open(jsonl_file, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.transform = transform or transforms.ToTensor()
        self.min_q = min_q
        self.max_q = max_q

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Keep as PIL so process_vision_info works later
        image_path = "../../AIML339/supplements_small/images/" + sample["image"]
        image = Image.open(image_path).convert("RGB")

        qa_pairs = sample["qas"]
        k = random.randint(self.min_q, min(self.max_q, len(qa_pairs)))
        chosen_pairs = random.sample(qa_pairs, k)

        questions_str = (
            user_prompt
            + "; ".join(f"Question: {p['q']} This corresponds to JSON key {p['k']}" for p in chosen_pairs)
            + "</QUESTIONS>"
        )
        answers_dict = {p['k']: p['a'] for p in chosen_pairs}
        answers_str = json.dumps(answers_dict, ensure_ascii=False)

        return {
            "image": image,  # PIL
            "questions": questions_str,
            "answers": answers_str
        }


# In[18]:


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                img = element.get("image", element)
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif not isinstance(img, Image.Image):
                    raise ValueError(f"Unsupported image type: {type(img)}")
                image_inputs.append(img)
    return image_inputs


# In[19]:


dataset_path = r'../../AIML339/supplements_small/output.jsonl'
dataset_obj = OCRVQADataset(dataset_path)


# In[20]:


import torch
from torch.utils.data import random_split

# Set split sizes
total_size = len(dataset_obj)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Use a generator with a manual seed for reproducibility
generator = torch.Generator().manual_seed(42)


# In[21]:


train_dataset, val_dataset, test_dataset = random_split(
    dataset_obj, [train_size, val_size, test_size], generator=generator
)

train_dataset_fmt = [format_data(sample) for sample in train_dataset]

print(train_dataset_fmt[100])


# In[32]:


from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

accelerator = Accelerator(mixed_precision="bf16")
writer = SummaryWriter(log_dir="./logs")

# Hugging Face model id
model_id = "google/gemma-3-4b-it"

# Check if GPU benefits from bfloat16
#if torch.cuda.get_device_capability()[0] < 8:
#    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Define model init arguments
#model_kwargs = dict(
#    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
#    torch_dtype=torch.bfloat16 # What torch dtype to use, defaults to auto
#)

#model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)


from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForImageTextToText

config = AutoConfig.from_pretrained(model_id)

with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config)

model = load_checkpoint_and_dispatch(
    model,
    model_id,
    device_map="auto",
    no_split_module_classes=["GemmaDecoderLayer"],  # or model-specific layers
    dtype=torch.bfloat16
)


model.config.max_position_embeddings = 1024

model.config.use_cache = False

processor = AutoProcessor.from_pretrained(base_model)


# In[31]:


print("Visible devices:", accelerator.state.num_processes)
print("Local device:", accelerator.device)


# In[25]:


from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

model = get_peft_model(model, peft_config)


# In[26]:


# Create a data collator to encode text and image pairs
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
        padding=True,
        truncation=True,
        max_length=512
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

    return batch  # keep on CPU, Accelerator will move to GPU


# In[27]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset_fmt,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)


# In[28]:

from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=2e-4)

#optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)


# In[29]:


model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)


# In[30]:


from torch.nn import CrossEntropyLoss

loss_fn = CrossEntropyLoss()
num_epochs = 1
gradient_accumulation_steps = 4

global_step = 0

model.train()
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else loss_fn(outputs.logits, batch["labels"])

        # Scale loss if using gradient accumulation
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Log to TensorBoard
            writer.add_scalar("train/loss", loss.item() * gradient_accumulation_steps, global_step)

    # Optional: save checkpoint at epoch end
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(f"gemma-supplements-small/epoch{epoch+1}")


# In[ ]:


accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("gemma-supplements-small/final")
writer.close()

