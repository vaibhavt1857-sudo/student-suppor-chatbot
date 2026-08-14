from flask import Blueprint, render_template, request, jsonify

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")

    # Simple demo logic (replace with NLP/ML later)
    if "hello" in user_message.lower():
        reply = "Hi there! How can I help you today?"
    elif "name" in user_message.lower():
        reply = "I’m Elementra, your AI assistant."
    else:
        reply = f"You asked: {user_message}. I’ll learn to answer smarter soon!"

    return jsonify({"reply": reply})

