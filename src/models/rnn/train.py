import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import os

from src.data_loader import load_text_data

# --- Configuration ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

# --- Model Definition ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        # Take the output of the last time step
        final_output = lstm_out[:, -1, :]
        return self.fc(final_output)

# --- Training Function ---
def train():
    """Trains the LSTM model and logs the experiment with MLflow."""
    print("Starting RNN model training...")

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test = load_text_data()

    vectorizer = CountVectorizer(max_features=5000)
    X_train_counts = vectorizer.fit_transform(X_train).toarray()
    X_test_counts = vectorizer.transform(X_test).toarray()

    vocab_size = len(vectorizer.vocabulary_)

    train_data = TensorDataset(torch.from_numpy(X_train_counts).long(), torch.from_numpy(y_train).long())
    test_data = TensorDataset(torch.from_numpy(X_test_counts).long(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 2. Initialize model, loss, and optimizer
    model = LSTMClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. MLflow setup
    mlflow.set_experiment("Transformer PoC")
    with mlflow.start_run(run_name="RNN_LSTM_Training") as run:
        mlflow.log_param("model_type", "RNN/LSTM")
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # 4. Training loop
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            for texts, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_epoch_loss:.4f}")
            mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)

        # 5. Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")
        mlflow.log_metric("test_accuracy", accuracy)

        # 6. Log model and plots
        mlflow.pytorch.log_model(model, "model")

        # Create and log loss plot
        plt.figure()
        plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("RNN/LSTM Training Loss")
        loss_plot_path = "training_loss_rnn.png"
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path, "plots")
        os.remove(loss_plot_path)

    print("RNN model training finished.")

if __name__ == "__main__":
    train()