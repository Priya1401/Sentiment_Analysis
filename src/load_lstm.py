import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define paths to your LSTM model and tokenizer
LSTM_MODEL_PATH = os.path.join("flask_sentiment_app", "model", "lstm","lstm_model.keras")
LSTM_TOKENIZER_PATH = os.path.join("flask_sentiment_app", "model","lstm", "tokenizer.pkl")

# Load the LSTM model
lstm_model = load_model(LSTM_MODEL_PATH)

# Load the LSTM tokenizer
with open(LSTM_TOKENIZER_PATH, "rb") as f:
    lstm_tokenizer = pickle.load(f)

def preprocess_lstm(text: str, max_len: int = 100):
    """Preprocess text for the LSTM model."""
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded