from flask import Flask, request, render_template_string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

# Sample FAQs
faq_data = {
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "How long does shipping take?": "Shipping usually takes 3-5 business days.",
    "Do you ship internationally?": "Yes, we ship internationally with applicable charges.",
    "How can I track my order?": "You can track your order using the tracking number sent to your email.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and Apple Pay."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Preprocess using SpaCy (lemmatization)
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# Response function
def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vec, X)
    best_match = np.argmax(similarity)

    if similarity[0][best_match] < 0.3:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    return answers[best_match]

# Flask Routes
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FAQ Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            height: 100vh;
        }
        .container {
            background: white;
            margin-top: 60px;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 500px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            display: flex;
            margin-top: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 16px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin-left: 10px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }
        .chat-box {
            margin-top: 30px;
        }
        .chat-entry {
            background: #f1f1f1;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .chat-entry.user {
            background-color: #d1ecf1;
        }
        .chat-entry.bot {
            background-color: #d4edda;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>FAQ Chatbot</h1>
        <form method="POST">
            <input type="text" name="user_input" placeholder="Ask a question..." required>
            <input type="submit" value="Ask">
        </form>
        {% if response %}
        <div class="chat-box">
            <div class="chat-entry user">
                <strong>You asked:</strong><br>{{ user_input }}
            </div>
            <div class="chat-entry bot">
                <strong>Bot response:</strong><br>{{ response }}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    response = ''
    user_input = ''
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = get_response(user_input)
    return render_template_string(HTML_TEMPLATE, response=response, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
