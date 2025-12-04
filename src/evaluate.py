"""
Evaluate a saved model on the MNIST test set without retraining.

Usage:
    python -m src.evaluate --model linear_svm
"""

import argparse
import joblib
import sys
from sklearn.metrics import accuracy_score, confusion_matrix

# Internal imports
from .utils.paths import DATA_DIR, MODEL_DIR
from .utils.data_loading import load_mnist
from .metrics import summarize_confusions

def _load_saved_model(model_name: str):
    """
    Load a trained model from the models/ directory.
    """
    model_path = MODEL_DIR / f"{model_name}.joblib"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print(f"Have you trained it yet? Run: python -m src.train --model {model_name}")
        sys.exit(1)
        
    print(f"Loading model: {model_path}")
    return joblib.load(model_path)

def evaluate_model(model_name: str):
    pipeline = _load_saved_model(model_name)

    print(f"Loading MNIST data from {DATA_DIR}...")
    _, _, X_test, y_test = load_mnist(DATA_DIR)
    
    print(f"Running inference on {len(X_test)} test samples...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("-" * 40)
    print(f"Model: {model_name}")
    print(f"Test Set Accuracy: {acc:.4f}")
    print("-" * 40)
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST model.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the saved model (e.g., linear_svm, xgboost)"
    )
    
    args = parser.parse_args()
    evaluate_model(args.model)

if __name__ == "__main__":
    main()
