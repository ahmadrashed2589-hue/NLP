from pydoc import text

import gradio as gr
import sys
import os

sys.path.append(os.path.abspath("."))

from src.predict import predict_bert, predict_baseline

def predict(text, model_type):
    if model_type == "Baseline (Logistic Regression)":
        emoji, confidence = predict_baseline(text)
    else:
        if not os.path.exists("models/bert_model"):
            return "❌ الخطأ: النموذج غير موجود", "يرجى تدريب BERT أولاً (python src/bert_model.py)"
        emoji, confidence = predict_bert(text)
    return emoji, f"{confidence:.2f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=3, label="Enter your text (أدخل النص هنا)"),
        gr.Radio(
            choices=["Baseline (Logistic Regression)", "Deep Learning (BERT)"], 
            value="Baseline (Logistic Regression)", 
            label="Select Model (اختر النموذج)"
        )
    ],
    outputs=[
        gr.Textbox(label="Predicted Emoji (الإيموجي المتوقع)"),
        gr.Textbox(label="Confidence (نسبة الثقة)")
    ],
    title="Emoji Prediction from Text 😃",
    description="Enter a short sentence or tweet, and the model will predict the most suitable emoji.\nأدخل جملة قصيرة أو تغريدة، وسيقوم النموذج بتوقع الإيموجي الأنسب."
)

if __name__ == "__main__":
    demo.launch()