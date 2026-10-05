import os
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# FIX 1: Simplify client initialization for the free Serverless API
client = InferenceClient(api_key=HF_TOKEN)

# Recommended model for free text-generation inference
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

def generate_flashcards(topic):
    if not topic or not topic.strip():
        return "Please enter a topic first."

    prompt = f"""Create 5 useful study flashcards about: {topic}

Use exactly this format:

Q1: question
A1: answer

Q2: question
A2: answer

Q3: question
A3: answer

Q4: question
A4: answer

Q5: question
A5: answer

Keep questions simple and answers short.
"""

    try:
        # FIX 2: Use the standard chat completion API for Meta Llama models
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface Setup
with gr.Blocks(title="AI Flashcard Generator") as app:
    gr.Markdown(
        """
        # 📚 AI Flashcard Generator
        ### Generate study flashcards using Hugging Face AI
        """
    )

    topic = gr.Textbox(
        label="Enter Topic",
        placeholder="Example: Stack in Data Structures"
    )

    generate_button = gr.Button("🚀 Generate Flashcards")

    output = gr.Textbox(
        label="Generated Flashcards",
        lines=15
    )

    generate_button.click(
        fn=generate_flashcards,
        inputs=topic,
        outputs=output
    )

if __name__ == "__main__":
    app.launch()
