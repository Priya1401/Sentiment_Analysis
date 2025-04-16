"""
lstm.py
--------------
This script performs the following steps:
  1. Loads and cleans the Tweets.csv dataset.
  2. Encodes the sentiment labels.
  3. Tokenizes the tweet text and pads the sequences.
  4. Performs hyperparameter tuning with cross-validation.
  5. Builds, trains, and evaluates an improved LSTM model with tuned parameters.
  6. Saves the best trained model and tokenizer in the 'flask_sentiment_app/model' directory.
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from itertools import product
from tensorflow.keras.layers import Input


# Data Cleaning Function
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

def build_model(max_vocab, max_len, embed_dim, lstm_units, dropout_rate, l2_reg, learning_rate, num_classes):
    """
    Build LSTM model with given hyperparameters.
    """
    model = Sequential()
    model.add(Input(shape=(max_len,)))  # Explicit Input layer
    model.add(Embedding(input_dim=max_vocab,
                        output_dim=embed_dim,
                        embeddings_regularizer=l2(l2_reg)))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units[0], return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(lstm_units[1])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def main():
    # 1. Load Data
    print("Loading Tweets.csv dataset...")
    df = pd.read_csv("Tweets.csv")  # Ensure Tweets.csv is accessible; update the path if needed
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    if "text" not in df.columns or "airline_sentiment" not in df.columns:
        raise ValueError("CSV must have 'text' and 'airline_sentiment' columns.")

    # 2. Clean Text
    print("Cleaning tweet texts...")
    df["cleaned_text"] = df["text"].apply(clean_text)
    print("Sample cleaned text:", df["cleaned_text"].head(1).values[0])

    # 3. Encode Sentiment Labels
    print("Encoding sentiment labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["airline_sentiment"])  # 0: negative, 1: neutral, 2: positive
    num_classes = len(label_encoder.classes_)
    y_cat = to_categorical(y_encoded, num_classes=num_classes)
    print("Unique sentiment labels:", label_encoder.classes_)

    # 4. Tokenize Text and Pad Sequences
    print("Tokenizing and padding text sequences...")
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])
    sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")
    print("Example padded sequence:", padded_sequences[0])

    # 5. Compute Class Weights
    class_counts = np.bincount(y_encoded)
    total_samples = len(y_encoded)
    class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(class_counts)}
    print("Computed class weights:", class_weights)

    # 6. Hyperparameter Grid
    param_grid = {
        'max_vocab': [10000, 15000],
        'max_len': [100, 150],
        'embed_dim': [128, 256],
        'lstm_units': [(128, 64), (256, 128)],
        'dropout_rate': [0.3, 0.5],
        'l2_reg': [1e-4, 1e-3],
        'learning_rate': [1e-3, 5e-4],
        'batch_size': [32, 64]
    }

    # 7. Cross-Validation and Hyperparameter Tuning
    print("Starting hyperparameter tuning...")
    best_score = 0
    best_params = {}
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    # Loop over all possible combinations
    for params in product(
        param_grid['max_vocab'],
        param_grid['max_len'],
        param_grid['embed_dim'],
        param_grid['lstm_units'],
        param_grid['dropout_rate'],
        param_grid['l2_reg'],
        param_grid['learning_rate'],
        param_grid['batch_size']
    ):
        max_vocab, max_len, embed_dim, lstm_units, dropout_rate, l2_reg, learning_rate, batch_size = params
        print(f"\nTesting parameter combination: {params}")

        # Adjust tokenizer and sequences for max_vocab and max_len
        tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
        tokenizer.fit_on_texts(df["cleaned_text"])
        sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

        scores = []
        fold_idx = 1
        for train_idx, val_idx in kfold.split(padded_sequences):
            print(f"  Starting fold {fold_idx}...")
            X_train, X_val = padded_sequences[train_idx], padded_sequences[val_idx]
            y_train, y_val = y_cat[train_idx], y_cat[val_idx]

            model = build_model(max_vocab, max_len, embed_dim, lstm_units, dropout_rate, l2_reg, learning_rate, num_classes)

            early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr],
                class_weight=class_weights,
                verbose=0
            )

            loss, acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"  Fold {fold_idx} validation accuracy: {acc:.4f}")
            scores.append(acc)
            fold_idx += 1

        mean_score = np.mean(scores)
        print(f"Mean CV Accuracy for combination {params}: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                'max_vocab': max_vocab,
                'max_len': max_len,
                'embed_dim': embed_dim,
                'lstm_units': lstm_units,
                'dropout_rate': dropout_rate,
                'l2_reg': l2_reg,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }

    print("\nHyperparameter tuning complete.")
    print(f"Best Params: {best_params}")
    print(f"Best CV Accuracy: {best_score:.4f}")

    # 8. Train Final Model with Best Parameters
    print("Preparing final model training using best parameters...")
    tokenizer = Tokenizer(num_words=best_params['max_vocab'], oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])
    sequences = tokenizer.texts_to_sequences(df["cleaned_text"])
    padded_sequences = pad_sequences(sequences, maxlen=best_params['max_len'], padding="post", truncating="post")
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, y_cat, test_size=0.2, stratify=y_encoded, random_state=42
    )

    final_model = build_model(
        best_params['max_vocab'],
        best_params['max_len'],
        best_params['embed_dim'],
        best_params['lstm_units'],
        best_params['dropout_rate'],
        best_params['l2_reg'],
        best_params['learning_rate'],
        num_classes
    )
    save_dir = os.path.join("../flask_sentiment_app", "model", "lstm")
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "lstm_model.keras")

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    print("Training final model...")
    history = final_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=best_params['batch_size'],
        callbacks=[early_stop, checkpoint, reduce_lr],
        class_weight=class_weights
    )
    print("Final model training complete.")

    # 9. Evaluate the Final Model
    loss, acc = final_model.evaluate(X_test, y_test)
    print(f"Final Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {acc:.4f}")

    # 10. Save Model and Tokenizer
    final_model.save(model_save_path, save_format='tf')
    print(f"Model saved to {model_save_path}")

    tokenizer_save_path = os.path.join(save_dir, "tokenizer.pkl")
    with open(tokenizer_save_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_save_path}")

if __name__ == "__main__":
    main()
