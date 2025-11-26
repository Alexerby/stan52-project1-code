"""
Test a trained MNIST classifier on custom PNG images.

This script loads a specified model from the `models/` directory and evaluates
it on hand-drawn digit images stored in `our-test-images/`. Each PNG image is
converted to grayscale, resized to 28×28 pixels, flattened to a 784-dimensional
vector, and passed through the model for prediction.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import joblib


def load_model(model_name: str):
    """
    Load a trained model by name from the `models/` directory.

    Parameters
    ----------
    model_name : str
        Base filename of the model without extension.

    Returns
    -------
    Any
        The deserialized scikit-learn model.
    """
    model_path = Path("models") / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    return joblib.load(model_path)


def load_custom_image(path: Path) -> np.ndarray:
    """
    Preprocess a custom PNG digit image for MNIST-style model input.

    Parameters
    ----------
    path : Path
        Filesystem path to the PNG image.

    Returns
    -------
    np.ndarray
        A (1, 784) array representing the flattened 28×28 grayscale image.
    """
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))

    arr = np.array(img, dtype=np.float32)

    # Uncomment if your images have white background and black digits:
    # arr = 255 - arr

    return arr.reshape(1, -1)


def test_model(model_name: str):
    """
    Run predictions on all images in `our-test-images/` using the specified model.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    """
    model = load_model(model_name)
    test_dir = Path("our-test-images")

    print("\nPredicting on custom images:\n")

    for img_path in sorted(test_dir.glob("*.png")):
        x = load_custom_image(img_path)
        pred = model.predict(x)[0]
        print(f"{img_path.name:10s} → predicted: {pred}")


def main():
    """Entry point for command-line use."""
    parser = argparse.ArgumentParser(
        description="Test MNIST models on custom digit images."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name (e.g., svm_rbf, random_forest, xgboost)."
    )

    args = parser.parse_args()
    test_model(args.model)


if __name__ == "__main__":
    main()
