import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocess import load_and_preprocess_data

def train_baseline():
    train_data, val_data, test_df, label_map, reverse_label_map = load_and_preprocess_data()
    
    os.makedirs("models", exist_ok=True)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train = vectorizer.fit_transform(train_data["clean_text"])
    X_val = vectorizer.transform(val_data["clean_text"])
    
    y_train = train_data["mapped_label"]
    y_val = val_data["mapped_label"]
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    joblib.dump(model, os.path.join("models", "logistic_model.pkl"))
    joblib.dump(vectorizer, os.path.join("models", "tfidf_vectorizer.pkl"))
    joblib.dump(reverse_label_map, os.path.join("models", "reverse_label_map.pkl"))

if __name__ == "__main__":
    train_baseline()