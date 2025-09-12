import os

class PromptLoader:
    def __init__(self):
        self.current_dir = os.path.dirname(__file__)
        self.prompts_path = os.path.join(self.current_dir, 'prompts')

    def get_prompt(self, prompt_name: str) -> str:
        file_path = os.path.join(self.prompts_path, f"{prompt_name}.txt")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Prompt '{prompt_name}' not found at {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()