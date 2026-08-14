import os
import json
import torch
from transformers import DistilBertTokenizerFast
from train_distilbert_qa import DistilBertForQAAndIntent

def test_inference():
    model_dirs = [
        'models/distilbert_qa',
        'chatbot_project/models/distilbert_qa',
        'student-support-chatbot/models/distilbert_qa'
    ]

    model_dir = None
    for md in model_dirs:
        if os.path.exists(md):
            model_dir = md
            break

    if not model_dir:
        print(f"[ERROR] Trained model directory not found in {model_dirs}")
        return

    print(f"Loading trained Multi-Task DistilBERT model from {model_dir}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    intent_map_file = os.path.join(model_dir, 'intent_map.json')
    id2intent = {}
    if os.path.exists(intent_map_file):
        with open(intent_map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            id2intent = {int(k): v for k, v in data.get('id2intent', {}).items()}

    num_intents = len(id2intent) if id2intent else 13
    model = DistilBertForQAAndIntent.from_pretrained(model_dir, num_intent_labels=num_intents)
    model.eval()

    print("\n" + "=" * 60)
    print("1. Testing Question Answering (Extractive QA)")
    print("=" * 60)

    question = "What is Results-Based Accountability?"
    context = (
        "Results-Based Accountability (also known as RBA) is a disciplined way of thinking "
        "and taking action that communities can use to improve the lives of children, youth, "
        "families, adults and the community as a whole."
    )

    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs['start_logits'])
    end_idx = torch.argmax(outputs['end_logits'])

    predict_tokens = inputs.input_ids[0][start_idx : end_idx + 1]
    predicted_answer = tokenizer.decode(predict_tokens, skip_special_tokens=True)

    print(f"Question       : {question}")
    print(f"Context        : {context[:80]}...")
    print(f"Predicted Span : '{predicted_answer}'")

    print("\n" + "=" * 60)
    print("2. Testing Intent Classification")
    print("=" * 60)

    test_queries = [
        "hello there",
        "who are you",
        "when are the exams",
        "what is the fee structure",
        "how to contact student support",
        "where is the library located"
    ]

    for query in test_queries:
        inputs = tokenizer(query, "", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_intent_id = int(torch.argmax(outputs['intent_logits']))
        intent_tag = id2intent.get(predicted_intent_id, f"ID_{predicted_intent_id}")
        print(f"Query: '{query:<35}' -> Predicted Intent: {intent_tag}")

    print("=" * 60)
    print("Multi-Task DistilBERT Model Verification Successful!")

if __name__ == '__main__':
    test_inference()
