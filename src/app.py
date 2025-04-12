import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

# Import BERT model, tokenizer, and preprocessing function
# from load_bert import bert_model, bert_tokenizer, preprocess_bert

# Import LSTM model and preprocessing function
from load_lstm import lstm_model, preprocess_lstm

# Define sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

# Create the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_choice = "LSTM"  # default
    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        model_choice = request.form.get("model_choice", "LSTM")  # Capture user's selection
        if tweet:
            if model_choice == "LSTM":
                processed_input = preprocess_lstm(tweet)
                pred = lstm_model.predict(processed_input)
                sentiment_idx = np.argmax(pred)
            elif model_choice == "BERT":
                from load_bert import bert_model, bert_tokenizer, preprocess_bert
                input_ids, attention_mask = preprocess_bert(tweet)
                pred = bert_model.predict([input_ids, attention_mask])
                # Adjust indexing if using logits:
                sentiment_idx = np.argmax(pred.logits) if hasattr(pred, "logits") else np.argmax(pred)
            # You might further map the index to the actual sentiment label here if desired.
            prediction = f"Predicted Sentiment: {LABELS[sentiment_idx]}"

    # Pass the model_choice back to the template so the form remembers the user's selection.
    return render_template("index.html", prediction=prediction, model_choice=model_choice)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)