import os
import pickle
import torch
import tkinter as tk
from tkinter import ttk
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# قاموس الرموز التعبيرية النهائي (20 كلاس)
emoji_map = {
    0: "😜", 1: "📸", 2: "😍", 3: "😂", 4: "😉",
    5: "🎄", 6: "📷", 7: "🔥", 8: "😘", 9: "❤",
    10: "😁", 11: "🇺🇸", 12: "☀", 13: "✨", 14: "💙",
    15: "💕", 16: "😎", 17: "😊", 18: "💜", 19: "💯"
}

# أسماء عائلات الماكرو للعرض والتوضيح
macro_names = {
    0: "Hearts Family (❤️ 😍 😘 💙 💕 💜)",
    1: "Joy & Laughs Family (😂 😜 😉 😁 😊)",
    2: "Energy & Cool Family (🔥 💯 😎)",
    3: "Photos & Weather Family (📸 📷 ☀ ✨)",
    4: "Events & Nations Family (🎄 🇺🇸)"
}

# 1. تحميل موديل الماكرو (Stage 1: BERT) من مسار مطلق آمن للويندوز
macro_path = os.path.abspath(os.path.join('models', 'macro_bert_model'))
macro_tokenizer, macro_model = None, None

if os.path.exists(macro_path):
    try:
        macro_tokenizer = BertTokenizer.from_pretrained(macro_path)
        macro_model = BertForSequenceClassification.from_pretrained(macro_path).to(device)
        macro_model.eval()
        print("✔ Stage 1: Macro BERT Model loaded successfully!")
    except Exception as e:
        print(f"Error loading Macro BERT: {e}")
else:
    print("❌ Error: Macro BERT model folder not found! Please run train_macro.py first.")

# 2. تحميل الموديلات الفرعية الخمسة مسبقاً في الذاكرة لتسريع الاستجابة (Stage 2: Micro Models)
micro_models = {}
micro_vectorizers = {}

for macro_id in range(5):
    model_file = f'models/micro_model_{macro_id}.pkl'
    vec_file = f'models/micro_vectorizer_{macro_id}.pkl'
    
    if os.path.exists(model_file) and os.path.exists(vec_file):
        with open(model_file, 'rb') as f:
            micro_models[macro_id] = pickle.load(f)
        with open(vec_file, 'rb') as f:
            micro_vectorizers[macro_id] = pickle.load(f)
print("✔ Stage 2: All 5 Micro Models loaded successfully into memory!")


def run_hierarchical_prediction():
    custom_text = text_entry.get()
    
    if not custom_text.strip():
        result_label.config(text="Please enter text first!", fg="#dc3545", font=("Arial", 12, "bold"))
        macro_info_label.config(text="")
        return
        
    clean_text = custom_text.lower().strip()
    
    if macro_model is None:
        result_label.config(text="Macro BERT model not loaded!", font=("Arial", 12), fg="red")
        return

    # --- المرحلة الأولى: توقع العائلة الكبرى عبر BERT ---
    inputs = macro_tokenizer(
        [clean_text],
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = macro_model(**inputs)
        macro_pred = torch.argmax(outputs.logits, dim=1).item()
    
    # تحديث واجهة المستخدم لتبين العائلة المقدرة (قوة المعمارية الهرمية)
    detected_family = macro_names.get(macro_pred, "Unknown Family")
    macro_info_label.config(text=f"Detected: {detected_family}", fg="#28a745")

    # --- المرحلة الثانية: توجيه النص للموديل الفرعي المخصص الحافظ للأوزان ---
    if macro_pred in micro_models and macro_pred in micro_vectorizers:
        vectorizer = micro_vectorizers[macro_pred]
        model = micro_models[macro_pred]
        
        # تحويل النص وتحذير الكلاس النهائي
        text_vec = vectorizer.transform([clean_text])
        final_class_idx = model.predict(text_vec)[0]
        
        # جلب الإيموجي النهائي وطباعته بحجم كبير واحترافي
        actual_emoji = emoji_map.get(int(final_class_idx), "❓")
        result_label.config(text=actual_emoji, font=("Segoe UI Emoji", 64))
    else:
        result_label.config(text="Micro model for this family missing!", font=("Arial", 12), fg="red")


def paste_action(event=None):
    try:
        text_entry.insert(tk.INSERT, root.clipboard_get())
    except tk.TclError:
        pass
    return "break"

# حل مشكلة حرف "ر" واللصق بالعربي عبر الـ Keycode للويندوز
def global_control_v_handler(event):
    if event.keycode == 86:  # كود مفتاح V / ر الثابت نظاماً
        return paste_action()

def show_context_menu(event):
    context_menu.tk_popup(event.x_root, event.y_root)


# --- بناء واجهة التطبيق المثالية للمناقشة ---
root = tk.Tk()
root.title("Hierarchical Emoji Predictor")
root.geometry("520x420")
root.configure(bg="#f8f9fa")

# العناوين الرئيسية
tk.Label(root, text="Hierarchical AI Emoji Predictor", font=("Arial", 16, "bold"), bg="#f8f9fa", fg="#007bff").pack(pady=15)
tk.Label(root, text="Enter Text to Analyze Emotion:", font=("Arial", 11), bg="#f8f9fa", fg="#495057").pack(pady=5)

# حقل إدخال الجمل
text_entry = tk.Entry(root, width=48, font=("Arial", 12), bd=2, relief="groove")
text_entry.pack(pady=5)

# ربط أحداث اللصق الآمنة
text_entry.bind("<Control-v>", paste_action)
text_entry.bind("<Control-V>", paste_action)
text_entry.bind("<Control-KeyPress>", global_control_v_handler)

context_menu = tk.Menu(root, tearoff=0)
context_menu.add_command(label="Paste (لصق)", command=paste_action)
text_entry.bind("<Button-3>", show_context_menu)

# زر التحليل والتنبؤ
predict_btn = tk.Button(root, text="Analyze & Predict", command=run_hierarchical_prediction, font=("Arial", 11, "bold"), bg="#007bff", fg="white", width=20, height=1, relief="flat", activebackground="#0056b3")
predict_btn.pack(pady=15)

# ملصق يوضح الموديل في أي مرحلة الحين (ماكرو)
macro_info_label = tk.Label(root, text="", font=("Arial", 11, "italic"), bg="#f8f9fa")
macro_info_label.pack(pady=5)

# ملصق النتيجة النهائية للإيموجي الكبير
result_label = tk.Label(root, text="Result Area", font=("Arial", 12, "bold"), bg="#f8f9fa", fg="#6c757d")
result_label.pack(pady=15)

root.mainloop()