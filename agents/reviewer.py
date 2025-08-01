# agents/reviewer.py
from transformers import pipeline

class ReviewerAgent:
    def __init__(self, model_name="Qwen/Qwen2-7B-Instruct"):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def review(self, code):
        prompt = f"""
Review this Python code for:
- Bugs
- Performance
- Readability
- Best practices

Code:
{code}

Provide a clear, constructive review.
        """
        result = self.pipe(prompt, max_new_tokens=512)
        return result[0]['generated_text']
