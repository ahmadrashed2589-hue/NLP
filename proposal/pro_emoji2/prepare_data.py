import pandas as pd
import re
import emoji
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove @user mentions
    text = re.sub(r'@user', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove hashtags symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove emojis to prevent data leakage (so the model learns words, not shapes)
    text = emoji.replace_emoji(text, replace='')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Loading data...")
df = pd.read_csv('data/Train.csv')
print(f"Original shape: {df.shape}")

print("Cleaning text...")
df['TEXT'] = df['TEXT'].apply(clean_text)

# Drop empty texts
df = df[df['TEXT'] != ""]

print("Applying undersampling to majority classes...")
MAX_SAMPLES = 3000
balanced_dfs = []

for label in df['Label'].unique():
    subset = df[df['Label'] == label]
    if len(subset) > MAX_SAMPLES:
        # Undersample randomly
        subset = subset.sample(n=MAX_SAMPLES, random_state=42)
    balanced_dfs.append(subset)

balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"New shape after undersampling: {balanced_df.shape}")
print("New class distribution:")
print(balanced_df['Label'].value_counts())

balanced_df.to_csv('data/Clean_Train.csv', index=False)
print("Saved cleaned and balanced data to data/Clean_Train.csv")
