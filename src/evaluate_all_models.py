#!/usr/bin/env python
"""
evaluate_all_models.py

This script evaluates all sentiment analysis models (traditional and deep learning)
on the test set from Tweets.csv and saves the evaluation visualizations (confusion matrix
plots and classification reports) using evaluation_visualizations module.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import evaluation visualization functions
from evaluation_visualizations import evaluate_model

# Import traditional model prediction functions
from train_models.logistic_regression import predict_sentiment as predict_logistic
from train_models.random_forest import predict_sentiment_rf as predict_rf
from train_models.svm import predict_sentiment_svm as predict_svm

# Import deep learning models and their preprocessing functions
from load_lstm import lstm_model, preprocess_lstm
from load_bert import bert_model, preprocess_bert

# Import data processing functions (to load dataset)
from data_processing import load_dataset

# Define sentiment labels; ensure these match your dataset's encoding.
LABELS = ["negative", "neutral", "positive"]




def evaluate_traditional_model(predict_func, test_texts, y_true, model_name, dir):
    """
    Evaluate a traditional model (Logistic, Random Forest, or SVM).
    It calls the provided prediction function on each tweet in test_texts.
    The prediction function is expected to return a sentiment label as a string.
    """
    y_pred = []
    MODEL_DIR = os.path.join("flask_sentiment_app", "model", dir)
    for tweet in test_texts:
        pred_label = predict_func(tweet, MODEL_DIR)
        try:
            pred_index = LABELS.index(pred_label)
        except ValueError:
            pred_index = -1  # If the predicted label is not valid
        y_pred.append(pred_index)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    evaluate_model(y_true, y_pred, LABELS, model_name)


def evaluate_deep_model(model, preprocess_func, test_texts, y_true, model_name, model_type="LSTM"):
    """
    Evaluate a deep learning model. For each tweet, use the corresponding preprocessing function.
    For LSTM, preprocess_lstm returns the input ready for model.predict.
    For BERT, preprocess_bert returns [input_ids, attention_mask].
    """
    y_pred = []
    for tweet in test_texts:
        if model_type == "LSTM":
            processed_input = preprocess_func(tweet)
            pred = model.predict(processed_input)
            pred_index = int(np.argmax(pred))
        elif model_type == "BERT":
            input_ids, attention_mask = preprocess_func(tweet)
            # Depending on your load_bert implementation, the prediction might return an object
            # containing logits or raw predictions.
            pred = model.predict([input_ids, attention_mask])
            pred_index = int(np.argmax(pred.logits) if hasattr(pred, "logits") else np.argmax(pred))
        else:
            pred_index = -1
        y_pred.append(pred_index)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    evaluate_model(y_true, y_pred, LABELS, model_name)


def main():
    # Load the dataset (assumes Tweets.csv has a column 'text' and the cleaned text is in 'cleaned_text')
    df = load_dataset("Tweets.csv")
    # Use the cleaned text and original sentiment labels; assume sentiment labels are strings matching LABELS.
    texts = df['cleaned_text'].tolist()
    # Map labels to indices using the order in LABELS.
    print(df['airline_sentiment'])
    y = [LABELS.index(label) for label in df['airline_sentiment']]

    # Split the dataset; we'll use only the test split for evaluation.
    _, X_test, _, y_test = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)

    print("Evaluating Logistic Regression model...")
    evaluate_traditional_model(predict_logistic, X_test, y_test, "Logistic Regression", "logistic")

    print("Evaluating Random Forest model...")
    evaluate_traditional_model(predict_rf, X_test, y_test, "Random Forest", "rf")

    print("Evaluating SVM model...")
    evaluate_traditional_model(predict_svm, X_test, y_test, "SVM", "svm")

    print("Evaluating LSTM model...")
    evaluate_deep_model(lstm_model, preprocess_lstm, X_test, y_test, "LSTM", model_type="LSTM")

    print("Evaluating BERT model...")
    evaluate_deep_model(bert_model, preprocess_bert, X_test, y_test, "BERT", model_type="BERT")


if __name__ == "__main__":
    main()
