import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def load_text_data(file_path="data/raw/synthetic_text_data.csv"):
    """Loads text data from a CSV file and splits it into training and testing sets."""
    print(f"Loading text data from {file_path}...")
    df = pd.read_csv(file_path)
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Text data loaded and split successfully.")
    return X_train, X_test, y_train, y_test

def get_image_datasets(directory_path="data/raw/images"):
    """Loads image data and creates training and testing datasets."""
    print(f"Loading image data from {directory_path}...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=directory_path, transform=transform)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print("Image datasets created successfully.")
    return train_dataset, test_dataset

if __name__ == '__main__':
    # --- Test Text Data Loader ---
    X_train, X_test, y_train, y_test = load_text_data()
    print(f"Text Train Samples: {len(X_train)}")
    print(f"Text Test Samples: {len(X_test)}")

    # --- Test Image Data Loader ---
    train_dataset, test_dataset = get_image_datasets()
    print(f"Image Train Samples: {len(train_dataset)}")
    print(f"Image Test Samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # You can iterate through the loaders to get batches of data
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)
    #     break
