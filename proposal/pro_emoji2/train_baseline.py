import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

print("Loading cleaned data...")
df = pd.read_csv('data/Clean_Train.csv')
df = df.dropna(subset=['TEXT', 'Label'])

X = df['TEXT']
y = df['Label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression with class_weight='balanced'...")
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
model.fit(X_train_vec, y_train)

print("Evaluating...")
y_pred = model.predict(X_test_vec)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Weighted F1-Score: {f1:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

import os

print("Saving models...")
os.makedirs('models', exist_ok=True)
with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Recreate reverse_label_map from classes
reverse_label_map = {i: i for i in range(20)}
with open('models/reverse_label_map.pkl', 'wb') as f:
    pickle.dump(reverse_label_map, f)

print("Done!")
