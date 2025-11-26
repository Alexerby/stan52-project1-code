"""
Train a selected machine-learning model on MNIST.

Usage:
    python train_mnist.py --model linear_svm
    python train_mnist.py --list-models
"""

import argparse
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.paths import DATA_DIR, MODEL_DIR
from src.preprocessor import load_mnist

# ---------------------------------------------------------------------------


def make_pipeline(spec: dict) -> Pipeline:
    """
    Build a Pipeline from a model specification.
    """
    steps = []
    if spec["scaler"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec["model"]))
    return Pipeline(steps)


# ---------------------------------------------------------------------------

def train_and_evaluate(model_name: str, spec: dict,
                       X_train, y_train, X_test, y_test):
    """
    Train one model and print evaluation metrics.
    """
    print(f"Training model: {model_name}")

    pipeline = make_pipeline(spec)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, out_path)

    print(f"\nAccuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"Saved model to: {out_path}\n")


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a MNIST model.")
    parser.add_argument(
        "--model",
        type=str,
        help="Name of model to train (see --list-models)."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # List models
    if args.list_models:
        print("Available models:")
        for name in MODEL_REGISTRY:
            print(" ", name)
        return

    # Check model selection
    if args.model is None:
        raise SystemExit("Error: --model is required unless using --list-models.")

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{args.model}'. "
            f"Run with --list-models to see available options."
        )

    # Load MNIST
    print(f"Loading MNIST from {DATA_DIR}...")
    X_train, y_train, X_test, y_test = load_mnist(DATA_DIR)
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

    # Train
    spec = MODEL_REGISTRY[args.model]
    train_and_evaluate(args.model, spec, X_train, y_train, X_test, y_test)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
