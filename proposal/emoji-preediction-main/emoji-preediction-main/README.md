# Emoji Prediction from Text using Deep Learning

This project predicts the most suitable emoji for a given text using NLP and machine learning.

## Dataset

The dataset contains tweets and emoji labels.  
The task is multi-class classification.

Files used:

- Train.csv
- Test.csv
- Mapping.csv
- OutputFormat.csv

## Models

### Baseline Model
- TF-IDF vectorization
- Logistic Regression classifier

### Deep Learning Model
- DistilBERT Transformer model
- Classification layer for emoji prediction

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

## How to Run

Install requirements:

```bash
pip install -r requirements.txt