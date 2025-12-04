"""
Evaluate a saved model on the MNIST test set without retraining.

Usage:
    python -m src.evaluate --model linear_svm
"""

import argparse
import joblib
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

from .utils.paths import DATA_DIR
from .utils.data_loading import load_mnist
from .metrics import summarize_confusions


def _load_saved_model(model_name: str, model_dir: Path):
    model_path = model_dir / f"{model_name}.joblib"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print(f"Have you trained it yet? Run: python3 -m src.training.train_mnist --model {model_name}")
        sys.exit(1)
        
    print(f"Loading model: {model_path}")
    return joblib.load(model_path)


def evaluate_model(model_name: str, model_base_dir: Path):
    print("-" * 40)
    print(f"Model: {model_name}")
    
    pipeline = _load_saved_model(model_name, model_base_dir)

    print(f"Loading MNIST data from {DATA_DIR}...")
    _, _, X_test, y_test = load_mnist(DATA_DIR)
    
    print(f"Running inference on {len(X_test)} test samples...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Confusion Matrix:")
    print(cm)

    print(f"Test Set Accuracy: {acc:.4f}")
    print("-" * 40)
    print(summarize_confusions(cm)) 

    
def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST model.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the saved model (e.g., linear_svm, xgboost)"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        required=False,
        help="Specify a custom directory where the model files are located."
    )
    
    args = parser.parse_args()
    
    default_model_dir = Path(__file__).resolve().parent.parent / "models"
    
    if args.dir:
        model_base_dir = Path(args.dir)
    else:
        model_base_dir = default_model_dir
        
    evaluate_model(args.model, model_base_dir)

if __name__ == "__main__":
    main()
