
# Airline Sentiment Analysis on Twitter

## Overview
This project implements a complete pipeline for sentiment analysis on the **Twitter US Airline Sentiment** dataset.  
It compares traditional machine‑learning methods (Logistic Regression, Random Forest, SVM) with deep‑learning approaches (LSTM, BERT), performs LDA topic modeling to uncover recurring themes, and exposes an interactive Flask web application for real‑time inference and model/topic exploration.

## Features
- **Data Preprocessing & Feature Engineering**
  - Cleans tweets (URL, mention, hashtag, punctuation removal; lowercase; tokenization; lemmatization)
  - Generates TF‑IDF vectors, Word2Vec embeddings, and BERT embeddings  
- **Traditional ML Models**   
  Logistic Regression | Random Forest | Support Vector Machine  
- **Deep‑Learning Models**   
  LSTM (Keras) | BERT (Hugging Face Transformers)  
- **LDA Topic Modeling**   
  Extracts key themes and ships an interactive **pyLDAvis** dashboard  
- **Evaluation & Visualization**
  - Classification reports + confusion‑matrix PNGs for every model
  - IEEE‑style tables for inclusion in the research paper
- **Flask Web App**
  - **Home** page: live sentiment prediction from any model  
  - **LDA Topics** page: embedded interactive topic‑model visualization  


## Installation
```bash
git clone https://github.com/yourusername/airline-sentiment-analysis.git
cd airline-sentiment-analysis

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

python - <<'PY'
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
PY
```

## Data
Download **Tweets.csv** from Kaggle (Twitter US Airline Sentiment) and place it in the project root.  
Required columns: `text`, `airline_sentiment`.

## Preprocessing
`data_processing.py` exposes helpers to
- cleanse raw tweets
- create `cleaned_text`
- generate `X_tfidf`, `X_w2v`, and BERT embeddings

## Training Models
```bash
python logistic_regression.py   # trains & saves model/logreg.pkl
python random_forest.py         # trains & saves model/rf.pkl
python svm.py                   # trains & saves model/svm.pkl
python lstm.py                  # trains & saves model/lstm_model.h5
python bert.py                  # fine‑tunes & saves model/bert_model/
```

## Evaluation
Generate classification reports + confusion matrices:
```bash
python evaluate_all_models.py   # PNGs saved in static/evaluations/
```

## LDA Topic Modeling
```bash
python lda_topic_modeling.py    # outputs lda_vis.html
mv lda_vis.html static/lda_vis.html
```

## Running the Flask App
```bash
python app.py
```
- **Home**: <http://localhost:5000/>  
- **LDA Topics**: <http://localhost:5000/lda>
