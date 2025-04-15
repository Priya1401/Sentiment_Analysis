#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_processing.py

This module isolates data processing and feature engineering tasks for the sentiment analysis project.
It includes:
  • Text cleaning using regular expressions and NLTK (with an alternate implementation using spaCy)
  • Dataset loading and tweet text cleaning
  • Feature generation using TF-IDF
  • Training a Word2Vec model and computing average tweet embeddings
  • Generating BERT-based embeddings
  • Label encoding for sentiment labels
"""

import re
import nltk
import pandas as pd
import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download required NLTK data (only the first time)
# nltk.download()
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean input text by:
      - Removing URLs, Twitter mentions, and hashtags.
      - Removing non-alphabetic characters.
      - Converting text to lowercase.
      - Tokenizing, filtering stopwords, and lemmatizing tokens.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\@[\w]*', '', text)
    text = re.sub(r'\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Alternative cleaning function using spaCy
import spacy

nlp = spacy.load("en_core_web_sm")


def clean_text_spacy(text):
    """
    Clean text using spaCy:
      - Converts text to lowercase.
      - Lemmatizes tokens.
      - Removes stopwords and non-alphabetic tokens.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


def load_dataset(csv_path='Tweets.csv'):
    """
    Load the tweet dataset and apply text cleaning.
    Expects a column 'text' in the CSV file.
    """
    df = pd.read_csv(csv_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df


# TF-IDF Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(texts, max_features=5000, ngram_range=(1, 2)):
    """
    Generate TF-IDF features from a collection of texts.
    Returns the TF-IDF feature matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer


# Word2Vec Feature Engineering
from gensim.models import Word2Vec


def train_word2vec_model(texts, vector_size=100, window=5, min_count=1, workers=4):
    """
    Train a Word2Vec model on tokenized texts.
    """
    sentences = [text.split() for text in texts]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model


def get_tweet_vector(tweet, w2v_model, vector_size=100):
    """
    Compute the average Word2Vec embedding for a tweet.
    Returns a zero vector if none of the tweet's words are in the model's vocabulary.
    """
    words = tweet.split()
    word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


def create_word2vec_features(texts, w2v_model, vector_size=100):
    """
    Generate Word2Vec-based feature vectors for a collection of texts.
    """
    features = np.array([get_tweet_vector(tweet, w2v_model, vector_size) for tweet in texts])
    return features


# BERT Embedding Feature Engineering
from transformers import BertTokenizer, BertModel
import torch

# Load BERT model and tokenizer (using base uncased model)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embedding(text, max_length=128):
    """
    Generate a BERT embedding for the given text using the CLS token representation.
    """
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Return the embedding for the [CLS] token
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


# Label Encoding for Sentiments
from sklearn.preprocessing import LabelEncoder


def encode_labels(labels):
    """
    Encode sentiment labels (e.g., Negative, Neutral, Positive) into numeric values.
    Returns the numeric labels and the fitted LabelEncoder.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder


# Main block for testing module functionality
if __name__ == "__main__":
    # Load dataset and show sample cleaned text
    df = load_dataset('Tweets.csv')
    print("Sample cleaned text:")
    print(df['cleaned_text'].head(), "\n")

    # Create TF-IDF features and print matrix shape
    X_tfidf, tfidf_vectorizer = create_tfidf_features(df['cleaned_text'])
    print("TF-IDF feature matrix shape:", X_tfidf.shape)

    # Train Word2Vec model and create Word2Vec features
    w2v_model = train_word2vec_model(df['cleaned_text'])
    X_w2v = create_word2vec_features(df['cleaned_text'], w2v_model)
    print("Word2Vec features shape:", X_w2v.shape)

    # Generate a BERT embedding for the first cleaned tweet and display its shape
    bert_embedding = get_bert_embedding(df['cleaned_text'].iloc[0])
    print("BERT embedding shape:", bert_embedding.shape)

    # Encode sentiment labels and show a sample of encoded values
    y_encoded, label_encoder = encode_labels(df['airline_sentiment'])
    print("Encoded labels (first five):", y_encoded[:5])
