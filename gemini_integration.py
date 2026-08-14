import os
import google.generativeai as genai
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import torch
from dotenv import load_dotenv

# -----------------------------
# 1. Configure Gemini API
# -----------------------------
# Load API key from environment (.env file)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=api_key)

# ✅ Use Gemini 3.6 Flash model
gemini_model = genai.GenerativeModel("gemini-3.6-flash")

# -----------------------------
# 2. Load DistilBERT QA model
# -----------------------------
# Use Hugging Face model instead of local heavy files
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# -----------------------------
# 3. DistilBERT QA Answer
# -----------------------------
def distilbert_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    )
    return answer

# -----------------------------
# 4. Gemini Answer
# -----------------------------
def gemini_answer(question):
    response = gemini_model.generate_content(question)
    return response.text

# -----------------------------
# 5. Routing Logic
# -----------------------------
def chatbot_response(user_query, context="Paris is the capital of France."):
    if len(user_query.split()) < 8:
        return distilbert_answer(user_query, context)
    else:
        return gemini_answer(user_query)

# -----------------------------
# 6. Test
# -----------------------------
if __name__ == "__main__":
    q1 = "What is the capital of France?"
    q2 = "Which city is the political center of France?"

    print("Direct:", chatbot_response(q1))
    print("Indirect:", chatbot_response(q2))
