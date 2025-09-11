import argparse
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --- Model Training Dispatcher ---
def main():
    """Parses the model_name argument and calls the appropriate training script."""
    parser = argparse.ArgumentParser(description="Main training script for the Transformer PoC project.")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=["rnn", "bert", "roberta", "distilbert", "vit"],
                        help="The name of the model to train.")
    args = parser.parse_args()

    model_name = args.model_name

    if model_name == "rnn":
        from src.models.rnn.train import train
        train()
    elif model_name == "bert":
        from src.models.bert.train import train
        train()
    elif model_name == "roberta":
        from src.models.roberta.train import train
        train()
    elif model_name == "distilbert":
        from src.models.distilbert.train import train
        train()
    elif model_name == "vit":
        from src.models.vit.train import train
        train()
    else:
        print(f"Error: Model '{model_name}' is not supported.")
        sys.exit(1)

if __name__ == "__main__":
    main()