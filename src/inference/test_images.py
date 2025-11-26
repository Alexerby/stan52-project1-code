"""
Run inference on custom PNG digit images using a saved model.

Usage:
    python test_images.py <model_name>
"""

from pathlib import Path
import argparse
import joblib
import numpy as np
from PIL import Image

from src.utils.paths import MODEL_DIR, TEST_IMAGE_DIR


# ---------------------------------------------------------------------------

def load_and_preprocess_image(path: Path) -> np.ndarray:
    """
    Load a PNG image and preprocess it into the MNIST 28x28 format.

    Returns
    -------
    ndarray of shape (784,)
    """
    img = Image.open(path).convert("L")  # grayscale

    # Resize to 28x28 if needed
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(img).astype(np.float32)

    # Optional: invert colors (uncomment if your custom images are black-on-white)
    # arr = 255 - arr

    arr = arr.reshape(-1)  # flatten
    return arr


def load_model(model_name: str):
    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}.\n"
            f"Available models: {[p.stem for p in MODEL_DIR.glob('*.joblib')]}"
        )
    return joblib.load(model_path)


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Test a saved MNIST model on PNG images.")
    parser.add_argument(
        "model",
        type=str,
        help="Name of the trained model (.joblib file) inside models/"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=TEST_IMAGE_DIR,
        help="Directory containing PNG images to classify."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model = load_model(args.model)

    image_paths = sorted(args.dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No PNG images found in {args.dir}")

    print(f"Loaded model: {args.model}")
    print(f"Found {len(image_paths)} images.\n")

    X = np.stack([load_and_preprocess_image(p) for p in image_paths])
    preds = model.predict(X)

    for path, pred in zip(image_paths, preds):
        print(f"{path.name:25s} → {pred}")

    print("\nDone.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
