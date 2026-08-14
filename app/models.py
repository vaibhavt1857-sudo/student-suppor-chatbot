import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example training data
questions = ["library timings", "hostel fee", "placement process"]
answers = ["Library open 9am-8pm", "Hostel fee is ₹30,000/year", "Placement starts in 7th semester"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
model = LogisticRegression()
model.fit(X, [0,1,2])

def get_answer(query):
    X_test = vectorizer.transform([query])
    pred = model.predict(X_test)[0]
    return answers[pred]
