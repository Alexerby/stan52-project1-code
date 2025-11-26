"""
Train a selected machine-learning model on MNIST.

This script loads MNIST from IDX files, builds a predefined model
pipeline, trains it, evaluates it on the MNIST test set, and saves
the fitted pipeline to ../models/.

Use:
    python train_mnist.py --model linear_svm
    python train_mnist.py --list-models
"""

from pathlib import Path
import argparse
import joblib
import preprocessor

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------

def load_mnist(data_dir: Path):
    """
    Load the MNIST dataset from IDX files.

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    X_train = preprocessor.load_idx_images(data_dir / "train-images.idx3-ubyte")
    y_train = preprocessor.load_idx_labels(data_dir / "train-labels.idx1-ubyte")
    X_test  = preprocessor.load_idx_images(data_dir / "t10k-images.idx3-ubyte")
    y_test  = preprocessor.load_idx_labels(data_dir / "t10k-labels.idx1-ubyte")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Registry of predefined models
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "linear_svm": {
        "scaler": False,
        "model": LinearSVC(dual=False)
    },
    "svm_rbf": {
        "scaler": True,
        "model": SVC(kernel="rbf", gamma="scale")
    },
    "svm_poly": {
        "scaler": True,
        "model": SVC(kernel="poly", degree=3)
    },
    "random_forest": {
        "scaler": False,
        "model": RandomForestClassifier(n_estimators=200)
    },
    "gradient_boosting": {
        "scaler": False,
        "model": GradientBoostingClassifier()
    },
    "xgboost": {
        "scaler": False,
        "model": XGBClassifier(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            subsample=0.8,
        )
    },
}


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
                       X_train, y_train, X_test, y_test,
                       output_dir: Path):
    """
    Train a single model and print evaluation results.
    """
    print(f"Training model: {model_name}")

    pipe = make_pipeline(spec)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(pipe, model_path)

    print(f"\nAccuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"Saved trained pipeline to: {model_path}\n")


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a MNIST model.")
    parser.add_argument(
        "--model",
        type=str,
        help="Model to train (see --list-models)."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/MNIST"),
        help="Path to MNIST IDX files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../models"),
        help="Directory to save trained pipelines."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:\n")
        for name in MODEL_REGISTRY:
            print("\t* ", name)
        return

    if args.model is None:
        raise SystemExit("Error: --model is required unless using --list-models.")

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{args.model}'. "
            f"Run with --list-models to see options."
        )

    print("Loading MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    print("Train set:", X_train.shape)
    print("Test set:",  X_test.shape)

    spec = MODEL_REGISTRY[args.model]
    train_and_evaluate(args.model, spec,
                       X_train, y_train, X_test, y_test,
                       args.output_dir)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
