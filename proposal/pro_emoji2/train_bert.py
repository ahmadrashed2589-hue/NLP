import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight

print("Checking CUDA...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading cleaned data...")
df = pd.read_csv('data/Clean_Train.csv')
df = df.dropna(subset=['TEXT', 'Label'])

X = df['TEXT'].tolist()
y = df['Label'].tolist()

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Computing class weights...")
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)

print("Tokenizing data...")
train_encodings = tokenize_function(X_train)
test_encodings = tokenize_function(X_test)

class EmojiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmojiDataset(train_encodings, y_train)
test_dataset = EmojiDataset(test_encodings, y_test)

print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(classes))
model.to(device)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Training BERT...")
trainer.train()

print("Saving model...")
os.makedirs('models/bert_model', exist_ok=True)
model.save_pretrained('models/bert_model')
tokenizer.save_pretrained('models/bert_model')

print("Done!")
