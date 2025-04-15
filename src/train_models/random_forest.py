#!/usr/bin/env python
# random_forest.py

import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.data_processing import load_dataset, create_tfidf_features, encode_labels, clean_text

# Define model saving directory
MODEL_DIR = os.path.join("..","flask_sentiment_app", "model","rf")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_random_forest(csv_path='../Tweets.csv'):
    # Load and preprocess the dataset
    df = load_dataset(csv_path)
    texts = df['cleaned_text']
    labels = df['airline_sentiment']

    # Generate TF-IDF features and encode labels
    X_tfidf, vectorizer = create_tfidf_features(texts)
    y, encoder = encode_labels(labels)

    # Split data (stratify to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model artifacts
    model_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    print("Random Forest model saved to:", model_path)


def predict_sentiment_rf(text, model_dir=MODEL_DIR):
    # Define file paths
    model_path = os.path.join(model_dir, 'rf_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

    # Load saved model artifacts
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Process text and predict
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    y_pred = model.predict(X)
    sentiment = encoder.inverse_transform(y_pred)[0]
    return sentiment


if __name__ == '__main__':
    train_random_forest()
