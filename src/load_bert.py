import os
from transformers import TFBertForSequenceClassification, BertTokenizer

# Define path to your BERT model
BERT_MODEL_PATH = os.path.join("flask_sentiment_app", "model", "bert_model")

# Load the BERT model
bert_model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)

# Load the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

def preprocess_bert(text: str, max_len: int = 128):
    """Preprocess text for the BERT model."""
    encoded = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf"
    )
    return encoded["input_ids"], encoded["attention_mask"]