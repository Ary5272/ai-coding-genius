# app.py
# 🤖 AI Coding Genius - Friendly, powerful, and ready to deploy
import os
import torch
import subprocess
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.io import wavfile
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset

# --- Load Code Model ---
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# --- Load TTS Model ---
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
embeds = load_dataset("Matthijs/cmu-arctic-xvectors", name="cmu_us_awb", split="validation")
speaker_embeddings = torch.tensor(embeds[7306]["xvector"]).unsqueeze(0)

# --- Safe Code Execution ---
def run_code_safely(code, timeout=10):
    os.makedirs("sandbox", exist_ok=True)
    path = "sandbox/temp_code.py"
    with open(path, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "❌ Execution timed out (infinite loop?)."
    except Exception as e:
        return "", f"💥 Error: {e}"

# --- Generate Code ---
def generate_code(prompt):
    if not prompt.strip():
        return "Hey! 😊 Please tell me what to code.", "Try something like 'bouncing ball'", None

    full_prompt = f"""
You're a friendly AI coder. Write clean, commented Python code.
Explain what you're doing.

Task: {prompt}
""".strip()

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.4)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    code = generated[len(full_prompt):].strip()

    # Extract if in ```python
    if "```python" in code:
        try:
            code = code.split("```python")[1].split("```")[0].strip()
        except:
            pass

    # Run code
    stdout, stderr = run_code_safely(code)
    output = f".Stdout:\n{stdout}\n.Stderr:\n{stderr}"

    # Generate voice
    voice_text = f"Here's your code for: {prompt[:60]}..."
    inputs_tts = tts_processor(text=voice_text, return_tensors="pt")
    speech = tts_model.generate_speech(inputs_tts["input_ids"], speaker_embeddings)
    wavfile.write("response.wav", 16000, speech.cpu().numpy())
    
    return code, output, "response.wav"

# --- Gradio UI ---
with gr.Blocks(title="AI Coding Genius") as demo:
    gr.Markdown("# 🤖 AI Coding Genius")
    gr.Markdown("I write and run Python code — from Pygame games to math art. Ask me anything!")

    with gr.Row():
        inp = gr.Textbox(label="💬 What should I code?", placeholder="e.g., A spinning 3D cube")
        btn = gr.Button("🚀 Generate & Run", variant="primary")

    code_out = gr.Code(label="💻 Generated Code")
    exec_out = gr.Textbox(label="📦 Execution Output")
    audio_out = gr.Audio(label="🎙️ Listen to Explanation")

    btn.click(generate_code, inp, [code_out, exec_out, audio_out])

demo.launch()
