import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# قاموس الرموز التعبيرية
emoji_map = {
    0: "😜", 1: "📸", 2: "😍", 3: "😂", 4: "😉",
    5: "🎄", 6: "📷", 7: "🔥", 8: "😘", 9: "❤",
    10: "😁", 11: "🇺🇸", 12: "☀", 13: "✨", 14: "💙",
    15: "💕", 16: "😎", 17: "😊", 18: "💜", 19: "💯"
}

model_path = os.path.abspath(os.path.join('models', 'distilbert_emoji_model'))
if not os.path.exists(model_path):
    print("❌ Error: Train the DistilBERT model first!")
    exit()

print("Loading data and simulating the exact Validation Split...")
df = pd.read_csv('data/Clean_Train.csv').dropna(subset=['TEXT', 'Label'])
X = df['TEXT'].tolist()
y = df['Label'].tolist()

# التقسيم المتطابق لضمان فحص بيانات لم يرها الموديل في التدريب
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Loading fine-tuned model and tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

print("\nAuditing predictions for Error Analysis...")
mismatches = []

# فحص عينة من بيانات التحقق (أول 1000 سطر لتوفير الوقت)
sample_size = min(1000, len(X_val))

with torch.no_grad():
    for i in tqdm(range(sample_size)):
        text = X_val[i]
        actual_label = y_val[i]
        
        clean_text = str(text).lower().strip()
        inputs = tokenizer([clean_text], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
        
        # إذا التوقع اختلف عن اللي حاطه البشر في الداتا
        if pred_label != actual_label:
            mismatches.append({
                'Text': text,
                'Actual': emoji_map.get(actual_label, "❓"),
                'Predicted': emoji_map.get(pred_label, "❓")
            })

# تحويل الأخطاء لإطار بيانات وعرض عينة ذهبية منها
mismatch_df = pd.DataFrame(mismatches)
mismatch_df.to_csv('data/Model_Errors_Analysis.csv', index=False)

print("\n" + "="*60)
print(f" Analysis Done! Found {len(mismatch_df)} mismatches out of {sample_size} checked sentences.")
print("Saved full error log to: 'data/Model_Errors_Analysis.csv'")
print("="*60)

print("\n🚀 Top 10 Interesting Contradictions (Sample):")
print(mismatch_df.head(10).to_string(index=False))