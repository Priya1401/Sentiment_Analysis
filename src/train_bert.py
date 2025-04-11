"""
train_bert.py
--------------
This script performs the following steps:
  1. Loads the Tweets.csv dataset.
  2. Encodes sentiment labels.
  3. Tokenizes the tweet text using BERT's tokenizer.
  4. Splits data into training and validation sets.
  5. Creates tf.data.Datasets for training and evaluation.
  6. Loads a pretrained BERT model for sequence classification (with 3 labels) and fine-tunes it.
  7. Trains the model using an optimizer and learning rate schedule.
  8. Evaluates the model on the validation set.
  9. Saves the trained model and tokenizer for later use in a Flask app.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer

# -------------------
# Hyperparameters
# -------------------
MAX_LENGTH = 128             # Maximum length of tokenized sequences
BATCH_SIZE = 32
EPOCHS = 3
INIT_LR = 2e-5
NUM_LABELS = 3               # For sentiments: Negative, Neutral, Positive

# -------------------
# 1. Load Data
# -------------------
df = pd.read_csv("Tweets.csv")  # Ensure Tweets.csv is accessible; update path if needed

if "airline_sentiment" not in df.columns or "text" not in df.columns:
    raise ValueError("CSV must have 'text' and 'airline_sentiment' columns.")

texts = df["text"].tolist()

# -------------------
# 2. Encode Sentiment Labels
# -------------------
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["airline_sentiment"])  # e.g., 0, 1, 2
labels = np.array(labels)  # Ensure it's a numpy array

# -------------------
# 3. Tokenize Text using BERT Tokenizer
# -------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="np"  # Return NumPy arrays for compatibility with train_test_split
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# -------------------
# 4. Split Data into Training and Validation Sets
# -------------------
train_ids, test_ids, train_mask, test_mask, train_labels, test_labels = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42, stratify=labels
)

# -------------------
# 5. Create tf.data Datasets
# -------------------
def create_dataset(ids, masks, labels):
    return tf.data.Dataset.from_tensor_slices((
        {"input_ids": ids, "attention_mask": masks},
        labels
    )).batch(BATCH_SIZE)

train_dataset = create_dataset(train_ids, train_mask, train_labels)
val_dataset = create_dataset(test_ids, test_mask, test_labels)

# -------------------
# 6. Load Pretrained BERT Model for Sequence Classification
# -------------------
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)

# Calculate total training steps for learning rate scheduling
steps_per_epoch = len(train_ids) // BATCH_SIZE
num_train_steps = steps_per_epoch * EPOCHS

optimizer, lr_schedule = create_optimizer(
    init_lr=INIT_LR,
    num_train_steps=num_train_steps,
    num_warmup_steps=0
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.summary()

# -------------------
# 7. Train the Model
# -------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# -------------------
# 8. Evaluate the Model
# -------------------
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# -------------------
# 9. Save the Model and Tokenizer
# -------------------
SAVE_DIR = os.path.join("flask_sentiment_app", "model", "bert_model")
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"BERT model and tokenizer saved to {SAVE_DIR}")
