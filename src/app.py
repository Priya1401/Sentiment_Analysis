"""
app.py
--------
Flask application for real-time tweet sentiment prediction.
Loads the trained LSTM model and tokenizer, processes user input,
and returns the sentiment prediction.
"""

import os
import pickle
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global constants
MAX_LEN = 100
labels = ["Negative", "Neutral", "Positive"]  # Ensure this order matches your label encoding

# Create the Flask app
app = Flask(__name__)

# Set the paths for the model and tokenizer
MODEL_PATH = os.path.join("flask_sentiment_app", "model", "lstm_model.h5")
TOKENIZER_PATH = os.path.join("flask_sentiment_app", "model", "tokenizer.pkl")

# Verify the files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")

# Load the model and tokenizer
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text: str):
    """Convert text to padded sequences using the loaded tokenizer."""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        if tweet:
            processed = preprocess_text(tweet)
            pred = model.predict(processed)
            sentiment_idx = np.argmax(pred)
            prediction = f"Predicted Sentiment: {labels[sentiment_idx]}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    # For development purposes; in production, use a proper WSGI server
    app.run(debug=True, host="0.0.0.0", port=5000)
