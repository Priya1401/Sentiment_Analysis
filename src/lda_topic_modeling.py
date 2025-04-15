#!/usr/bin/env python
# lda_topic_modeling.py

"""
LDA Topic Modeling for Tweet Sentiment Analysis

This module implements topic modeling using LDA on tweet data.
It performs the following steps:
  1. Loads the dataset from 'Tweets.csv'
  2. Preprocesses the text for LDA by tokenizing, lowercasing, removing stopwords,
     and filtering out tokens with less than 3 characters.
  3. Constructs a Gensim dictionary and corpus.
  4. Trains an LDA model with a specified number of topics.
  5. Prints the top words in each topic.
  6. Generates an interactive pyLDAvis visualization and saves it as an HTML file.

Make sure you have installed the required packages:
  pip install gensim pyLDAvis nltk pandas

Note: The code assumes the presence of either a 'cleaned_text' column in the CSV file,
or falls back to the raw 'text' column.
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure that NLTK stopwords are downloaded
nltk.download('stopwords')


def preprocess_for_lda(text):
    """
    Preprocess text for LDA:
      - Converts to lowercase.
      - Splits text into tokens.
      - Removes stopwords.
      - Filters out tokens with less than 3 characters.

    Args:
      text (str): The input text.

    Returns:
      list: A list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().split()
    return [word for word in tokens if word not in stop_words and len(word) > 2]


def train_lda(csv_path='Tweets.csv', num_topics=5, passes=10, output_html='lda_vis.html'):
    """
    Train an LDA model on tweet data and generate an interactive visualization.

    Args:
      csv_path (str): Path to the CSV file containing tweets.
      num_topics (int): Number of topics to extract.
      passes (int): Number of passes through the corpus during training.
      output_html (str): File path to save the interactive visualization.

    Returns:
      tuple: (lda_model, dictionary, corpus, vis) where:
          - lda_model: The trained Gensim LDA model.
          - dictionary: Gensim dictionary built from the dataset.
          - corpus: The corpus in bag-of-words format.
          - vis: The pyLDAvis visualization object.
    """
    # Load the dataset; prefer cleaned_text if available
    df = pd.read_csv(csv_path)
    if 'cleaned_text' in df.columns:
        texts = df['cleaned_text'].astype(str)
    else:
        texts = df['text'].astype(str)

    # Preprocess each tweet for LDA (each tweet becomes a list of tokens)
    processed_texts = texts.apply(preprocess_for_lda)

    # Create Gensim dictionary and corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Train the LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=42,
                         passes=passes,
                         alpha='auto',
                         per_word_topics=True)

    # Print the topics
    print("Topics found:")
    topics = lda_model.print_topics(num_words=5)
    for idx, topic in topics:
        print(f"Topic {idx}: {topic}")

    # Generate an interactive visualization using pyLDAvis and save as HTML
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_html)
    print(f"Interactive LDA visualization saved to {output_html}")

    return lda_model, dictionary, corpus, vis


if __name__ == "__main__":
    # Run the LDA training on the dataset and generate the visualization.
    train_lda()
