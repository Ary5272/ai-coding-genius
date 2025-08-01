# agents/coder.py
from transformers import AutoTokenizer, AutoModelForCausalLM

class CoderAgent:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-6.7b-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def generate(self, prompt):
        full_prompt = f"""
You're a brilliant, friendly AI coder. Explain clearly and write clean Python.
Include comments and use best practices.

Task: {prompt}
        """.strip()
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.4)
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code[len(full_prompt):].strip()
