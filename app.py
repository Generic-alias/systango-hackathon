from flask import Flask, request, jsonify
import joblib
import re
import string

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load('personality_model.pkl')  # Change filename if needed
vectorizer = joblib.load('vectorizer.pkl')  # Load the TF-IDF vectorizer

# Basic text preprocessing function

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()  # Tokenize by splitting on spaces
    stop_words = set(["the", "is", "in", "and", "to", "of", "a", "that", "it", "on", "for", "with", "as", "was", "at", "by", "an"])  # Minimal stopwords list
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    essay_text = data.get('essay', '')

    if not essay_text.strip():
        return jsonify({'error': 'No essay provided!'}), 400

    # Preprocess text
    cleaned_text = clean_text(essay_text)
    
    # Convert to numerical features using TF-IDF
    essay_tfidf = vectorizer.transform([cleaned_text])

    # Predict personality
    prediction = model.predict(essay_tfidf)

    return jsonify({'predicted_personality': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
