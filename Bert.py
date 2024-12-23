import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import classification_report

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and Preprocess the Dataset
def preprocess_data(texts, labels, tokenizer, max_length=128):
    tokenized_data = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return tokenized_data, torch.tensor(labels)

# Example: Replace this with your actual dataset
texts = ["Sample text 1", "Sample text 2", "Sample text 3"]  # Your text data
labels = [0, 1, 2]  # Replace with actual class labels (300 classes: 0-299)

# Split into training and validation sets
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings, train_labels = preprocess_data(texts_train, labels_train, tokenizer)
val_encodings, val_labels = preprocess_data(texts_val, labels_val, tokenizer)

# Create PyTorch Dataset
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}, self.labels[idx]

train_dataset = ClassificationDataset(train_encodings, train_labels)
val_dataset = ClassificationDataset(val_encodings, val_labels)

# 2. Initialize Model
num_classes = 300  # Update this with the number of classes
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
model.to(device)

# 3. Set Up Training
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 5  # Assuming 5 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

criterion = torch.nn.CrossEntropyLoss()

# 4. Training Loop
def train_model(model, train_loader, val_loader, optimizer, lr_scheduler, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss, correct_predictions = 0, 0

        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            correct_predictions += (outputs.logits.argmax(dim=-1) == labels).sum().item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {correct_predictions / len(train_loader.dataset):.4f}")

        # Validation
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    model.eval()
    val_loss, correct_predictions = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            correct_predictions += (outputs.logits.argmax(dim=-1) == labels).sum().item()

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {correct_predictions / len(val_loader.dataset):.4f}")

# Train the model
train_model(model, train_loader, val_loader, optimizer, lr_scheduler, epochs=5)

# 5. Save the Model
model.save_pretrained("bert_multi_class_model")
tokenizer.save_pretrained("bert_multi_class_model")

# 6. Evaluate the Model on Test Data
def evaluate_model(model, texts, labels):
    tokenized_data, test_labels = preprocess_data(texts, labels, tokenizer)
    test_dataset = ClassificationDataset(tokenized_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, zero_division=0))

# Example usage:
# evaluate_model(model, test_texts, test_labels)
