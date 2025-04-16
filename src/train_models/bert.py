"""
bert.py
--------------
This script performs the following steps:
  1. Loads and cleans the Tweets.csv dataset.
  2. Encodes sentiment labels and computes class weights.
  3. Tokenizes the tweet text using BERT's tokenizer.
  4. Performs hyperparameter tuning with k-fold cross-validation.
  5. Creates tf.data.Datasets for training and evaluation.
  6. Loads a pretrained BERT model, adds a custom classification head, and fine-tunes it.
  7. Trains the model with optimized hyperparameters.
  8. Evaluates the model on the validation set.
  9. Saves the best model and tokenizer for use in a Flask app.
Print statements are added for better visibility of execution progress.
"""

import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from itertools import product

# -------------------
# Hyperparameters
# -------------------
MAX_LENGTH = 150             # Increased to capture more context
EPOCHS = 5                   # Increased with early stopping
NUM_LABELS = 3               # Negative, Neutral, Positive

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

# -------------------
# Custom Model with Classification Head
# -------------------
def create_bert_model(pretrained_model_name, num_labels, dropout_rate=0.3):
    print(f"Creating BERT model with dropout_rate={dropout_rate}")
    # Load the pretrained model (TFBertForSequenceClassification already has a classifier head)
    bert = TFBertForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=num_labels
    )

    # Define input layers
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")

    # Wrap the BERT call inside a Lambda layer, explicitly converting inputs to tensors
    def bert_layer(inputs):
        input_ids, attention_mask = inputs
        # Convert to tf.Tensor in case they're KerasTensors.
        input_ids = tf.convert_to_tensor(input_ids)
        attention_mask = tf.convert_to_tensor(attention_mask)
        # Call the BERT model and return the logits for classification.
        outputs = bert({"input_ids": input_ids, "attention_mask": attention_mask}, training=False).logits
        return outputs

    # Specify the output shape: (batch_size, num_labels)
    lambda_output_shape = lambda input_shape: (input_shape[0][0], num_labels)

    bert_outputs = tf.keras.layers.Lambda(bert_layer, output_shape=lambda_output_shape)([input_ids, attention_mask])

    # Add a custom classification head on top of the BERT outputs.
    x = Dense(128, activation="relu")(bert_outputs)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_labels, activation="softmax")(x)

    # Create and return the final model.
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

# -------------------
# Create tf.data Dataset
# -------------------
def create_dataset(ids, masks, labels, batch_size):
    print(f"Creating dataset with batch_size={batch_size}, input_ids shape={ids.shape}")
    return tf.data.Dataset.from_tensor_slices((
        {"input_ids": ids, "attention_mask": masks},
        labels
    )).shuffle(1000).batch(batch_size)

# -------------------
# Main Function
# -------------------
def main():
    # 1. Load and Clean Data
    print("Loading Tweets.csv...")
    df = pd.read_csv("Tweets.csv")
    if "airline_sentiment" not in df.columns or "text" not in df.columns:
        print("ERROR: CSV must have 'text' and 'airline_sentiment' columns")
        raise ValueError("CSV must have 'text' and 'airline_sentiment' columns")

    df["cleaned_text"] = df["text"].apply(clean_text)
    texts = df["cleaned_text"].tolist()
    print(f"Loaded and cleaned {len(texts)} tweets")

    # 2. Encode Labels and Compute Class Weights
    print("Encoding sentiment labels and computing class weights...")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["airline_sentiment"])
    labels = np.array(labels)

    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = {i: total_samples / (NUM_LABELS * count) for i, count in enumerate(class_counts)}
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    # 3. Tokenize Text
    print("Tokenizing text with BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    print(f"Tokenized data: input_ids shape={input_ids.shape}, attention_mask shape={attention_mask.shape}")

    # 4. Hyperparameter Grid
    param_grid = {
        "init_lr": [2e-5, 3e-5],
        "batch_size": [16, 32],
        "dropout_rate": [0.3, 0.5]
    }
    print("Starting hyperparameter tuning with grid:", param_grid)

    # 5. K-Fold Cross-Validation for Hyperparameter Tuning
    best_score = 0
    # Initialize best_params with defaults
    best_params = {"init_lr": 2e-5, "batch_size": 16, "dropout_rate": 0.3}

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    for params in product(param_grid["init_lr"], param_grid["batch_size"], param_grid["dropout_rate"]):
        init_lr, batch_size, dropout_rate = params
        print(f"\nTesting hyperparameters: init_lr={init_lr}, batch_size={batch_size}, dropout_rate={dropout_rate}")

        scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(input_ids), 1):
            print(f"Processing fold {fold}/{kfold.n_splits}...")
            train_ids_fold, val_ids = input_ids[train_idx], input_ids[val_idx]
            train_mask_fold, val_mask = attention_mask[train_idx], attention_mask[val_idx]
            train_labels_fold, val_labels = labels[train_idx], labels[val_idx]

            train_dataset = create_dataset(train_ids_fold, train_mask_fold, train_labels_fold, batch_size)
            val_dataset = create_dataset(val_ids, val_mask, val_labels, batch_size)

            model = create_bert_model("bert-base-uncased", NUM_LABELS, dropout_rate)

            steps_per_epoch = len(train_ids_fold) // batch_size
            num_train_steps = steps_per_epoch * EPOCHS

            optimizer = AdamW(learning_rate=init_lr, weight_decay=0.01)
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"]
            )

            print(f"Training model for fold {fold}...")
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=EPOCHS,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6)
                ],
                verbose=1,
                class_weight=class_weights
            )

            _, acc = model.evaluate(val_dataset, verbose=0)
            print(f"Fold {fold} validation accuracy: {acc:.4f}")
            scores.append(acc)

        mean_score = np.mean(scores)
        print(f"Mean cross-validation accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = {"init_lr": init_lr, "batch_size": batch_size, "dropout_rate": dropout_rate}
            print(f"New best hyperparameters found: {best_params}, mean accuracy: {best_score:.4f}")

    print(f"\nHyperparameter tuning completed. Best hyperparameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    # 6. Create Final Train/Test Split and Train Final Model with Best Parameters
    print("Preparing final model training with train/test split...")
    train_ids, test_ids, train_mask, test_mask, train_labels, test_labels = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train set size: {len(train_ids)}, Test set size: {len(test_ids)}")

    train_dataset = create_dataset(train_ids, train_mask, train_labels, best_params["batch_size"])
    val_dataset = create_dataset(test_ids, test_mask, test_labels, best_params["batch_size"])

    final_model = create_bert_model("bert-base-uncased", NUM_LABELS, best_params["dropout_rate"])

    steps_per_epoch = len(train_ids) // best_params["batch_size"]
    num_train_steps = steps_per_epoch * EPOCHS

    optimizer = AdamW(learning_rate=best_params["init_lr"], weight_decay=0.01)
    final_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    save_dir = os.path.join("../flask_sentiment_app", "model", "bert_model")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, "best_model"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False
    )

    print(f"Starting final model training with {EPOCHS} epochs...")
    history = final_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            checkpoint
        ],
        verbose=1,
        class_weight=class_weights
    )

    # 7. Evaluate the Final Model
    print("\nEvaluating final model...")
    loss, accuracy = final_model.evaluate(val_dataset)
    print(f"Final validation loss: {loss:.4f}")
    print(f"Final validation accuracy: {accuracy:.4f}")

    # 8. Save the Model and Tokenizer
    print("\nSaving model and tokenizer...")
    final_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"BERT model and tokenizer saved to {save_dir}")

if __name__ == "__main__":
    main()
