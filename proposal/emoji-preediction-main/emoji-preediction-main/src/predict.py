import os
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import clean_text

def get_emoji_from_label(label_number, mapping_path="data/Mapping.csv"):
    mapping_df = pd.read_csv(mapping_path)
    row = mapping_df[mapping_df["number"] == label_number]
    if not row.empty:
        return row.iloc[0]["emoticons"]
    return ""

def predict_emoji(text: str, model_type: str = "baseline") -> dict:
    cleaned_text = clean_text(text)
    
    if model_type == "baseline":
        model = joblib.load(os.path.join("models", "logistic_model.pkl"))
        vectorizer = joblib.load(os.path.join("models", "tfidf_vectorizer.pkl"))
        reverse_label_map = joblib.load(os.path.join("models", "reverse_label_map.pkl"))
        
        vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = float(max(probabilities))
        
        original_label = reverse_label_map[prediction]
        emoji = get_emoji_from_label(original_label)
        
        return {"emoji": emoji, "confidence": confidence}
        
    elif model_type == "bert":
        reverse_label_map = joblib.load(os.path.join("models", "bert_reverse_label_map.pkl"))
        
        model_path = os.path.join("models", "bert_model")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.eval()
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
        original_label = reverse_label_map[prediction.item()]
        emoji = get_emoji_from_label(original_label)
        
        return {"emoji": emoji, "confidence": confidence.item()}
        
    return {}