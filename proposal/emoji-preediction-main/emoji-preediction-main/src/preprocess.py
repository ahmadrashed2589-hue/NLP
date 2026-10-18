import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

def load_and_preprocess_data(train_path="data/Train.csv", test_path="data/Test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df["clean_text"] = train_df["TEXT"].apply(clean_text)
    train_df["tokens"] = train_df["clean_text"].apply(tokenize)
    test_df["clean_text"] = test_df["TEXT"].apply(clean_text)
    test_df["tokens"] = test_df["clean_text"].apply(tokenize)
    
    top_10_labels = train_df["Label"].value_counts().nlargest(10).index
    train_df = train_df[train_df["Label"].isin(top_10_labels)].copy()
    
    label_map = {label: idx for idx, label in enumerate(top_10_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    train_df["mapped_label"] = train_df["Label"].map(label_map)
    
    train_data, val_data = train_test_split(
        train_df, test_size=0.10, random_state=42, stratify=train_df["mapped_label"]
    )
    
    return train_data, val_data, test_df, label_map, reverse_label_map