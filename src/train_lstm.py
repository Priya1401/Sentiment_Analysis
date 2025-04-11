"""
train_lstm.py
--------------
This script performs the following steps:
  1. Loads and cleans the Tweets.csv dataset.
  2. Encodes the sentiment labels.
  3. Tokenizes the tweet text and pads the sequences.
  4. Builds, trains, and evaluates an improved LSTM model.
     (with SpatialDropout, additional Dense layer, and regularization)
  5. Saves the trained model (lstm_model.h5) and tokenizer (tokenizer.pkl)
     in the 'flask_sentiment_app/model' directory.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -------------------
# Hyperparameters
# -------------------
MAX_VOCAB = 10000         # Maximum number of words in the vocabulary
MAX_LEN = 100             # Maximum length (number of tokens) for each tweet
EMBED_DIM = 128           # Dimension of the embedding vectors
BATCH_SIZE = 64
EPOCHS = 15               # Increase epochs; early stopping and callbacks will keep overfitting in check
L2_REG = 1e-4             # L2 regularization factor for the embedding layer

# -------------------
# Data Cleaning Function
# -------------------
def clean_text(text: str) -> str:
    """
    Remove URLs, mentions, hashtags, non-alphabetical characters,
    convert to lowercase, and strip extra spaces.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

def main():
    # -------------------
    # 1. Load Data
    # -------------------
    df = pd.read_csv("Tweets.csv")  # Ensure Tweets.csv is accessible; update the path if needed

    if "text" not in df.columns or "airline_sentiment" not in df.columns:
        raise ValueError("CSV must have 'text' and 'airline_sentiment' columns.")

    # -------------------
    # 2. Clean Text
    # -------------------
    df["cleaned_text"] = df["text"].apply(clean_text)

    # -------------------
    # 3. Encode Sentiment Labels
    # -------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["airline_sentiment"])  # 0: negative, 1: neutral, 2: positive
    num_classes = len(label_encoder.classes_)
    y_cat = to_categorical(y_encoded, num_classes=num_classes)

    # -------------------
    # 4. Tokenize Text and Pad Sequences
    # -------------------
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])
    sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # -------------------
    # 5. Train/Test Split
    # -------------------
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, y_cat, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # -------------------
    # 6. Build the Improved LSTM Model
    # -------------------
    model = Sequential()
    # Embedding layer with L2 regularization; using input_shape to build model immediately.
    model.add(Embedding(input_dim=MAX_VOCAB,
                        output_dim=EMBED_DIM,
                        input_shape=(MAX_LEN,),
                        embeddings_regularizer=l2(L2_REG)))
    # Use SpatialDropout to improve generalization on sequence data.
    model.add(SpatialDropout1D(0.2))
    # Stack a Bidirectional LSTM layer that returns sequences
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    # Another Bidirectional LSTM layer (final one without return_sequences)
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    # Additional Dense layer to capture deeper representations
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    # Final classification layer
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model with an appropriate loss and optimizer.
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Print model summary to verify that layers are built and parameter counts are correct.
    model.summary()

    # -------------------
    # 7. Set Up Callbacks for Better Training
    # -------------------
    save_dir = os.path.join("flask_sentiment_app", "model")
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "lstm_model.h5")

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    callbacks = [early_stop, checkpoint, reduce_lr]

    # -------------------
    # 8. Train the Model
    # -------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # -------------------
    # 9. Evaluate the Model
    # -------------------
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # -------------------
    # 10. Save Model and Tokenizer
    # -------------------
    save_dir = os.path.join("flask_sentiment_app", "model")
    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, "lstm_model.h5")
    tokenizer_save_path = os.path.join(save_dir, "tokenizer.pkl")

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    with open(tokenizer_save_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_save_path}")

if __name__ == "__main__":
    main()
