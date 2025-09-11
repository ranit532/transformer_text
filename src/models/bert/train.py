import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data_loader import load_text_data

# --- Configuration ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 128

# --- Simple word-level tokenizer ---
def simple_tokenizer(texts, max_length=MAX_LEN):
    vocab = {}
    tokenized = []
    for text in texts:
        tokens = text.lower().split()
        ids = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1
            ids.append(vocab[token])
        ids = ids[:max_length] + [0] * (max_length - len(ids))
        tokenized.append(ids)
    return torch.tensor(tokenized)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)
    def __getitem__(self, idx):
        return {'input_ids': self.encodings[idx], 'labels': self.labels[idx]}
    def __len__(self):
        return len(self.labels)

class SimpleClassifier(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_labels):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size + 1, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_labels)
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return type('Output', (), {'loss': loss, 'logits': logits})

# --- Training Function ---
def train():
    print("Starting BERT (custom) model training...")
    X_train, X_test, y_train, y_test = load_text_data()
    train_encodings = simple_tokenizer(list(X_train), max_length=MAX_LEN)
    test_encodings = simple_tokenizer(list(X_test), max_length=MAX_LEN)
    vocab_size = max([max(seq) for seq in train_encodings.tolist()])
    train_dataset = SimpleDataset(train_encodings, y_train)
    test_dataset = SimpleDataset(test_encodings, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = SimpleClassifier(vocab_size=vocab_size, hidden_dim=64, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mlflow.set_experiment("Transformer PoC")
    with mlflow.start_run(run_name="BERT_Custom") as run:
        mlflow.log_param("model_type", "Custom_BERT")
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_epoch_loss:.4f}")
            mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
                predictions = torch.argmax(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.pytorch.log_model(model, "model")
        plt.figure()
        plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Custom BERT Training Loss")
        loss_plot_path = "training_loss_bert_custom.png"
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path, "plots")
        os.remove(loss_plot_path)

if __name__ == "__main__":
    train()