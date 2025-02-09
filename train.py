import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset
from torch import cuda
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Set device to GPU if available
device = 'cuda' if cuda.is_available() else 'cpu'

# Load dataset
df_org = pd.read_csv("/content/7allV03.csv")
df_org = df_org.sample(frac=1.0, random_state=42)

# Get unique labels
labels = df_org['category'].unique().tolist()
labels = [s.strip() for s in labels]

NUM_LABELS = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# Map labels to integer ids
df_org["labels"] = df_org.category.map(lambda x: label2id[x.strip()])

# Load the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-uncased", max_length=512)
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(device)

# Split the data into train, validation, and test sets
SIZE = df_org.shape[0]
train_texts = list(df_org.text[:SIZE // 2])
val_texts = list(df_org.text[SIZE // 2:(3 * SIZE) // 4])
test_texts = list(df_org.text[(3 * SIZE) // 4:])
train_labels = list(df_org.labels[:SIZE // 2])
val_labels = list(df_org.labels[SIZE // 2:(3 * SIZE) // 4])
test_labels = list(df_org.labels[(3 * SIZE) // 4:])

# Function to tokenize data in batches to avoid memory issues
def batch_tokenize(texts, tokenizer, batch_size=32):
    encodings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_encodings = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors='pt')
        encodings.append(batch_encodings)
    # Merge all batch encodings into a single dictionary
    return {key: torch.cat([enc[key] for enc in encodings], dim=0) for key in encodings[0]}

# Tokenize the datasets in smaller batches
train_encodings = batch_tokenize(train_texts, tokenizer)
val_encodings = batch_tokenize(val_texts, tokenizer)
test_encodings = batch_tokenize(test_texts, tokenizer)

# Custom Dataset class for handling tokenized text data and corresponding labels
class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create DataLoader instances
train_dataloader = DataLoader(train_encodings, train_labels)
val_dataloader = DataLoader(val_encodings, val_labels)
test_dataset = DataLoader(test_encodings, test_labels)

# Define the function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

# Training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./TTC4900Model', 
    do_train=True,
    do_eval=True,
    num_train_epochs=1,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32,
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps", 
    fp16=True,
    load_best_model_at_end=True
)

# Initialize Trainer with model and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Evaluate on train, validation, and test sets
q = [trainer.evaluate(eval_dataset=df) for df in [train_dataloader, val_dataloader, test_dataset]]
print(pd.DataFrame(q, index=["train", "val", "test"]).iloc[:, :5])
print('trining end')
