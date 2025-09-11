import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os

from src.data_loader import get_image_datasets

# --- Configuration ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 16

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(16, num_classes)
    def forward(self, x, labels=None):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return type('Output', (), {'loss': loss, 'logits': logits})

# --- Training Function ---
def train():
    """Fine-tunes the Simple CNN model and logs the experiment with MLflow."""
    print("Starting Simple CNN model training...")

    # 1. Load data
    train_dataset, test_dataset = get_image_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. Load model and optimizer
    model = SimpleCNN(num_classes=2)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. MLflow setup
    mlflow.set_experiment("Transformer PoC")
    with mlflow.start_run(run_name="SimpleCNN") as run:
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # 4. Training loop
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, labels=labels)
                loss = outputs.loss
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
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")
        mlflow.log_metric("test_accuracy", accuracy)

        # 6. Log model and plots
        mlflow.pytorch.log_model(model, "model")

        plt.figure()
        plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Simple CNN Training Loss")
        loss_plot_path = "training_loss_simplecnn.png"
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path, "plots")
        os.remove(loss_plot_path)

    print("Simple CNN model training finished.")

if __name__ == "__main__":
    train()
