import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for

# Import the LSTM model and its preprocessing function
from load_lstm import lstm_model, preprocess_lstm

# Define sentiment labels for deep models
LABELS = ["Negative", "Neutral", "Positive"]

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    # Default model selection. You can extend your HTML select options with these values.
    model_choice = "LSTM"
    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()
        # The model_choice should be provided via the form (e.g., "LSTM", "BERT", "Logistic", "RandomForest", "SVM")
        model_choice = request.form.get("model_choice", "LSTM")
        if tweet:
            if model_choice == "LSTM":
                # Preprocess tweet using the LSTM preprocessing function
                processed_input = preprocess_lstm(tweet)
                # Predict using the LSTM model
                pred = lstm_model.predict(processed_input)
                sentiment_idx = np.argmax(pred)
                prediction = f"Predicted Sentiment: {LABELS[sentiment_idx]}"
            elif model_choice == "BERT":
                # Import BERT model, tokenizer, and preprocessing function on-demand
                from load_bert import bert_model, bert_tokenizer, preprocess_bert
                # Preprocess the tweet using the BERT preprocessing function
                input_ids, attention_mask = preprocess_bert(tweet)
                pred = bert_model.predict([input_ids, attention_mask])
                sentiment_idx = np.argmax(pred.logits) if hasattr(pred, "logits") else np.argmax(pred)
                prediction = f"Predicted Sentiment: {LABELS[sentiment_idx]}"
            elif model_choice == "Logistic":
                # Import the logistic regression prediction function
                from train_models.logistic_regression import predict_sentiment as predict_logistic
                MODEL_DIR = os.path.join("flask_sentiment_app", "model", "logistic")
                sentiment = predict_logistic(tweet, MODEL_DIR)
                prediction = f"Predicted Sentiment: {sentiment}"
            elif model_choice == "RandomForest":
                # Import the random forest prediction function
                from train_models.random_forest import predict_sentiment_rf as predict_rf
                MODEL_DIR = os.path.join("flask_sentiment_app", "model", "rf")
                sentiment = predict_rf(tweet, MODEL_DIR)
                prediction = f"Predicted Sentiment: {sentiment}"
            elif model_choice == "SVM":
                # Import the SVM prediction function
                from train_models.svm import predict_sentiment_svm as predict_svm
                MODEL_DIR = os.path.join("flask_sentiment_app", "model", "svm")
                sentiment = predict_svm(tweet, MODEL_DIR)
                prediction = f"Predicted Sentiment: {sentiment}"
            else:
                prediction = "Invalid model selection."

    return render_template("index.html", prediction=prediction, model_choice=model_choice)

@app.route("/lda")
def lda_topics():
    """
    This route renders the LDA Topics dashboard.
    It assumes that your LDA visualization (lda_vis.html) is placed in the static folder.
    """
    return render_template("lda_dashboard.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
