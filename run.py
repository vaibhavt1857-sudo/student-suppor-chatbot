import os
import json
import torch
import uuid
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
from train_distilbert_qa import DistilBertForQAAndIntent
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

app = Flask(__name__)

# -----------------------------
# 1. Configure Gemini API (Text)
# -----------------------------
load_dotenv()  # Load variables from .env file
api_key = os.getenv("GOOGLE_API_KEY")   # Read from environment

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-3.6-flash")
print("Gemini API key loaded successfully.")

def gemini_answer(question):
    response = gemini_model.generate_content(question)
    return response.text

# -----------------------------
# 2. Load intents definition file
# -----------------------------
intents_path = "data/intents.json"
if not os.path.exists(intents_path):
    intents_path = "student-support-chatbot/data/intents.json"

with open(intents_path, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

tag2responses = {item["tag"]: item.get("responses", ["How can I assist you?"]) for item in intents_data.get("intents", [])}

# -----------------------------
# 3. Load DistilBERT Model & Tokenizer from Hugging Face
# -----------------------------
print("Loading DistilBERT model from Hugging Face...")

# Use pretrained DistilBERT QA model
model_name = "distilbert-base-uncased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# If you have custom intent labels, set them here
num_intents = 13
model = DistilBertForQAAndIntent.from_pretrained(model_name, num_intent_labels=num_intents)
model.eval()

# -----------------------------
# 4. Flask Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    user_message = data.get("message", "").strip().lower()
    context_text = data.get("context", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a valid message or question."})

    if user_message in ["hi", "hello", "hey"]:
        return jsonify({"reply": "Hello! I am Elementra, your AI assistant, developed by VAIBHAV KUMAR TIWARI.", "type": "identity"})

    if user_message in ["who are you", "what are you", "introduce yourself"]:
        return jsonify({"reply": "I am Elementra, proudly developed by VAIBHAV KUMAR TIWARI.", "type": "identity"})

    if "who built you" in user_message:
        return jsonify({"reply": "I was built by VAIBHAV KUMAR TIWARI.", "type": "identity"})

    if "your name" in user_message:
        return jsonify({"reply": "My name is Elementra.", "type": "identity"})

    if context_text:
        inputs = tokenizer(user_message, context_text, return_tensors="pt", max_length=192, truncation="longest_first")
        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = torch.argmax(outputs["start_logits"])
        end_idx = torch.argmax(outputs["end_logits"])
        if end_idx >= start_idx:
            predict_tokens = inputs.input_ids[0][start_idx : end_idx + 1]
            predicted_answer = tokenizer.decode(predict_tokens, skip_special_tokens=True).strip()
            if predicted_answer:
                return jsonify({"reply": predicted_answer, "type": "qa"})

    gemini_reply = gemini_answer(user_message)
    developer_name = "VAIBHAV KUMAR TIWARI"
    for word in ["Gemini", "Google", "Google Gemini", "Gemini AI"]:
        gemini_reply = gemini_reply.replace(word, developer_name)

    return jsonify({"reply": gemini_reply, "type": "gemini"})

# -----------------------------
# Navigation Page Routes
# -----------------------------
@app.route("/about-us")
@app.route("/about")
def about_us():
    return render_template("about.html")

@app.route("/features")
@app.route("/ai-features")
def features():
    return render_template("features.html")

@app.route("/join-us")
@app.route("/join")
def join_us():
    return render_template("join.html")

@app.route("/news")
def news():
    return render_template("news.html")

# -----------------------------
# 5. Image Generation Route (Diffusers)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

pipe = pipe.to(device)
pipe.enable_attention_slicing()

@app.route("/generate_image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "")

    try:
        # Faster defaults: fewer steps + smaller resolution
        image = pipe(prompt, num_inference_steps=25, height=384, width=384).images[0]
        image_path = os.path.join("static", f"generated_{uuid.uuid4().hex}.png")
        image.save(image_path)
        return jsonify({"image_url": f"/{image_path}"})
    except Exception as e:
        print("Diffusers image generation error:", e)
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 6. Run Flask App
# -----------------------------
if __name__ == "__main__":
    print("Flask app starting on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
