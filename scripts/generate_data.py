import os
import csv
from faker import Faker
from PIL import Image
import random

# --- Configuration ---
NUM_TEXT_SAMPLES = 1000
NUM_IMAGE_SAMPLES_PER_CLASS = 100
IMAGE_SIZE = (64, 64)

RAW_DATA_DIR = "data/raw"
TEXT_DATA_FILE = os.path.join(RAW_DATA_DIR, "synthetic_text_data.csv")
IMAGE_DATA_DIR = os.path.join(RAW_DATA_DIR, "images")

# --- Text Data Generation ---
def generate_text_data():
    """Generates a CSV file with synthetic text data for sentiment analysis."""
    print("Generating synthetic text data...")
    fake = Faker()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    with open(TEXT_DATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for _ in range(NUM_TEXT_SAMPLES):
            label = random.choice([0, 1])
            if label == 1:
                # Positive sentiment
                text = fake.sentence(ext_word_list=['good', 'great', 'awesome', 'excellent', 'love', 'happy'])
            else:
                # Negative sentiment
                text = fake.sentence(ext_word_list=['bad', 'terrible', 'awful', 'hate', 'sad', 'poor'])
            writer.writerow([text, label])

    print(f"Successfully generated {NUM_TEXT_SAMPLES} text samples in {TEXT_DATA_FILE}")

# --- Image Data Generation ---
def generate_image_data():
    """Generates synthetic image data for classification."""
    print("Generating synthetic image data...")

    os.makedirs(IMAGE_DATA_DIR, exist_ok=True)

    for i in range(2): # Two classes: 0 and 1
        class_dir = os.path.join(IMAGE_DATA_DIR, f"class_{i}")
        os.makedirs(class_dir, exist_ok=True)

        for j in range(NUM_IMAGE_SAMPLES_PER_CLASS):
            # Generate a simple image with a random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new("RGB", IMAGE_SIZE, color=color)
            img.save(os.path.join(class_dir, f"image_{j}.png"))

    print(f"Successfully generated {NUM_IMAGE_SAMPLES_PER_CLASS * 2} image samples in {IMAGE_DATA_DIR}")

if __name__ == "__main__":
    generate_text_data()
    generate_image_data()