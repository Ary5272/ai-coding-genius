# app.py
# 🤖 AI Coding Genius: Omega Core
# "The most advanced AI coder deployable on Hugging Face"
import os
import torch
import gradio as gr
import numpy as np
from scipy.io import wavfile
from sentence_transformers import SentenceTransformer
import faiss
import zipfile
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 🔌 Import Local Modules
# ─────────────────────────────────────────────────────────────
try:
    from agents.coder import CoderAgent
    from agents.reviewer import ReviewerAgent
    from utils.tts import TextToSpeech
    from utils.stt import SpeechToText
    from utils.preview import render_pygame_preview
    from utils.export import create_project_zip
    from utils.gist import create_gist
except Exception as e:
    gr.Error(f"Missing module: {e}")

# ─────────────────────────────────────────────────────────────
# 🧠 Initialize Systems
# ─────────────────────────────────────────────────────────────
# Coder
coder = CoderAgent()

# Reviewer
reviewer = ReviewerAgent()

# TTS
tts = TextToSpeech()

# STT
stt = SpeechToText()

# Memory (FAISS)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
memory_store = []  # Stores {"prompt": str, "code": str}

# Stable Diffusion (Texture Gen)
try:
    from diffusers import StableDiffusionPipeline
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    sd_ready = True
except:
    sd_ready = False

# ─────────────────────────────────────────────────────────────
# 🛠️ Helper: Run Code Safely
# ─────────────────────────────────────────────────────────────
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
        return "", "Timeout: Code took too long."
    except Exception as e:
        return "", f"Error: {e}"

# ─────────────────────────────────────────────────────────────
# 💬 Main Coding Pipeline
# ─────────────────────────────────────────────────────────────
def full_pipeline(prompt):
    if not prompt.strip():
        return "Say something!", None, None, None

    # 1. Generate code
    raw_code = coder.generate(prompt)
    final_code = raw_code
    if "```python" in raw_code:
        try:
            final_code = raw_code.split("```python")[1].split("```")[0].strip()
        except:
            pass

    # 2. Review code
    review = reviewer.review(final_code)

    # 3. Run code
    stdout, stderr = run_code_safely(final_code)
    output = f".Stdout:\n{stdout}\n.Stderr:\n{stderr}"

    # 4. Save to memory
    embedding = embedder.encode(prompt)
    index.add(np.array([embedding]))
    memory_store.append({"prompt": prompt, "code": final_code})

    # 5. Voice explanation
    voice_msg = f"Here's your code for {prompt[:60]}..."
    audio_path = tts.speak(voice_msg, "assets/response.wav")

    # 6. Try preview (if Pygame)
    preview_img = None
    if "pygame" in final_code.lower():
        buf = render_pygame_preview(final_code)
        if buf:
            preview_img = Image.open(buf)

    return final_code, output, audio_path, review, preview_img

# ─────────────────────────────────────────────────────────────
# 🎨 Generate Texture with SD
# ─────────────────────────────────────────────────────────────
def generate_texture(prompt):
    if not sd_ready:
        return None
    try:
        image = sd_pipe(prompt, num_inference_steps=30).images[0]
        return image
    except:
        return None

# ─────────────────────────────────────────────────────────────
# 📦 Export Project
# ─────────────────────────────────────────────────────────────
def export_project(code, include_texture=None):
    assets = {}
    if include_texture:
        assets["texture.png"] = b"dummy"  # In real, save image
    zip_buffer = create_project_zip(code, assets)
    return zip_buffer

# ─────────────────────────────────────────────────────────────
# 🎙️ Voice to Code
# ─────────────────────────────────────────────────────────────
def voice_to_code(audio):
    text = stt.transcribe(audio)
    return text  # Feed into main pipeline

# ─────────────────────────────────────────────────────────────
# 🧠 Show Memory
# ─────────────────────────────────────────────────────────────
def show_memory():
    if not memory_store:
        return "No past projects."
    return "\n\n".join([
        f"🔹 {m['prompt'][:50]}..." for m in memory_store[-5:]
    ])

# ─────────────────────────────────────────────────────────────
# 🚀 Gradio UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="AI Coding Genius: Omega") as demo:
    gr.Markdown("# 🤯 AI Coding Genius: Omega Edition")
    gr.Markdown("The most advanced AI coder on Hugging Face. It **talks**, **codes**, **reviews**, **remembers**, and **creates**.")

    with gr.Tabs():
        # ─────────────────────────────────────────────────────
        with gr.Tab("💻 Code & Run"):
            inp = gr.Textbox(label="💬 What should I code?", placeholder="e.g., A rotating 3D cube")
            btn = gr.Button("🚀 Generate & Run", variant="primary")

            code_out = gr.Code(label="Generated Code")
            output_out = gr.Textbox(label="Output")
            audio_out = gr.Audio(label="🎙️ AI Voice")
            review_out = gr.Textbox(label="🔍 AI Review")
            preview_out = gr.Image(label="🎮 Preview (if Pygame)")

            btn.click(
                full_pipeline,
                inp,
                [code_out, output_out, audio_out, review_out, preview_out]
            )

        # ─────────────────────────────────────────────────────
        with gr.Tab("🎙️ Voice Input"):
            mic = gr.Audio(source="microphone", type="filepath", label="Speak your coding task")
            mic_btn = gr.Button("🎤 Transcribe & Code")
            mic_out = gr.Textbox(label="Transcribed Prompt")

            mic_btn.click(voice_to_code, mic, mic_out)

        # ─────────────────────────────────────────────────────
        with gr.Tab("🎨 Generate Texture"):
            tex_prompt = gr.Textbox(label="Describe texture", placeholder="e.g., cyberpunk city wall")
            tex_btn = gr.Button("Generate")
            tex_out = gr.Image(label="Generated Texture")

            tex_btn.click(generate_texture, tex_prompt, tex_out)

        # ─────────────────────────────────────────────────────
        with gr.Tab("📦 Export Project"):
            exp_btn = gr.Button("Download ZIP")
            exp_out = gr.File(label="Project ZIP")

            exp_btn.click(
                lambda code: gr.File(value=export_project(code), visible=True),
                inputs=code_out,
                outputs=exp_out
            )

        # ─────────────────────────────────────────────────────
        with gr.Tab("🧠 Memory"):
            mem_btn = gr.Button("Show Recent Projects")
            mem_out = gr.Textbox(label="Your AI's Memory")
            mem_btn.click(show_memory, None, mem_out)

# ─────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
