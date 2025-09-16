from torch.utils.data import Dataset
from PIL import Image
import json, random
from torchvision import transforms
import os
from prompt_loader import PromptLoader
import numpy as np

prompts = PromptLoader()
system_message = prompts.get_prompt("system")
user_prompt = prompts.get_prompt("user")

data_dir = os.getenv('DATA_DIR') # where to load the images/jsonl file from
dataset_path = os.path.join(data_dir, "output_cleaned.jsonl")
images_path  = os.path.join(data_dir, "images")

class OCRVQADataset(Dataset):
    def __init__(self, jsonl_file, transform=None, min_q=1, max_q=4):
        with open(jsonl_file, 'r') as f:
            self.samples = [json.loads(line) for line in f]

        self.transform = transform or transforms.ToTensor()
        self.min_q = min_q
        self.max_q = max_q

        # Filter invalid upfront (instead of returning None later)
        import numpy as np
        self.samples = [
            s for s in self.samples if any(
                p.get("a") not in (None, "", float("nan"))
                and not (isinstance(p.get("a"), float) and np.isnan(p["a"]))
                for p in s["qas"]
            )
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = os.path.join(images_path, sample["image"])
        image = Image.open(image_path).convert("RGB")

        qa_pairs = [
            p for p in sample["qas"]
            if p.get("a") not in (None, "", float("nan"))
            and not (isinstance(p.get("a"), float) and np.isnan(p["a"]))
        ]

        k = random.randint(self.min_q, min(self.max_q, len(qa_pairs)))
        chosen_pairs = random.sample(qa_pairs, k)

        questions_str = (
            user_prompt
            + "; ".join(
                f'Question: "{prompts.get_prompt(p["q"])}" '
                f'This corresponds to JSON key "{p["k"]}"'
                for p in chosen_pairs
            )
            + "</QUESTIONS>"
        )
        answers_dict = {p['k']: p['a'] for p in chosen_pairs}
        answers_str = json.dumps(answers_dict, ensure_ascii=False)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": questions_str},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answers_str}],
                },
            ]
        }

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


