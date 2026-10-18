import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# إعداد كرت الشاشة
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# قاموس الرموز التعبيرية الـ 20 الكامل لمشروعك
emoji_map = {
    0: "😜", 1: "📸", 2: "😍", 3: "😂", 4: "😉",
    5: "🎄", 6: "📷", 7: "🔥", 8: "😘", 9: "❤",
    10: "😁", 11: "🇺🇸", 12: "☀", 13: "✨", 14: "💙",
    15: "💕", 16: "😎", 17: "😊", 18: "💜", 19: "💯"
}

model_path = os.path.abspath(os.path.join('models', 'distilbert_emoji_model'))

if not os.path.exists(model_path):
    print(f"❌ Error: Model folder not found at {model_path}")
    exit()

print("Loading your fine-tuned DistilBERT model into GPU...")
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # وضع الموديل في طور التقييم والتنبؤ

print("\n" + "="*50)
print("🚀 DistilBERT Emoji Predictor is READY!")
print("Type your text and press Enter. (Type 'exit' to quit)")
print("="*50 + "\n")

while True:
    user_text = input("Enter Text: ")
    
    if user_text.lower().strip() == 'exit':
        print("Goodbye!")
        break
        
    if not user_text.strip():
        continue
        
    # تنظيف وتجهيز النص
    clean_text = user_text.lower().strip()
    
    # تحويل النص إلى تيرمز (Tokenization)
    inputs = tokenizer(
        [clean_text],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)
    
    # التنبؤ الصامت بدون حساب Gradients لتسريع العملية
    with torch.no_grad():
        outputs = model(**inputs)
        # جلب الكلاس صاحب أعلى نسبة احتمالية
        prediction = torch.argmax(outputs.logits, dim=1).item()
        
    # جلب الإيموجي المقابل من القاموس
    predicted_emoji = emoji_map.get(prediction, "❓")
    
    print(f"Predicted Emoji: {predicted_emoji}\n")