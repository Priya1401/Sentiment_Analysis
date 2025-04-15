#!/usr/bin/env python
# svm_model.py

import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from src.data_processing import load_dataset, create_tfidf_features, encode_labels, clean_text

# Define model saving directory
MODEL_DIR = os.path.join("..","flask_sentiment_app", "model","svm")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_svm(csv_path='../Tweets.csv'):
    # Load dataset and preprocess texts and labels
    df = load_dataset(csv_path)
    texts = df['cleaned_text']
    labels = df['airline_sentiment']

    # Generate TF-IDF features and encode labels
    X_tfidf, vectorizer = create_tfidf_features(texts)
    y, encoder = encode_labels(labels)

    # Split the dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train an SVM using LinearSVC
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save the SVM model, vectorizer, and encoder
    model_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    print("SVM model saved to:", model_path)


def predict_sentiment_svm(text, model_dir=MODEL_DIR):
    # Define file paths
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

    # Load saved artifacts
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Process input text and predict
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    y_pred = model.predict(X)
    sentiment = encoder.inverse_transform(y_pred)[0]
    return sentiment


if __name__ == '__main__':
    train_svm()