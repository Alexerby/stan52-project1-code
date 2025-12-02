## Overview

This project trains a few machine-learning models on the MNIST dataset and then tests them on some hand-drawn PNG digit images.
Models are trained using scikit-learn and XGBoost and saved as `.joblib` files so they can be loaded later for inference.

Workflow:

1. Load MNIST from IDX files
2. Train a chosen model
3. Save it in `models/`
4. Test it on PNG images in `our-test-images/`

---

## Project Structure

```
project/
│
├── data/
│   └── MNIST/                 # Raw IDX files
│
├── models/                    # Saved trained models
│
├── our-test-images/           # Custom PNG digits
│
└── src/
    ├── utils/                 # paths.py (centralized paths)
    ├── preprocessing/         # MNIST IDX loading helpers
    ├── training/              # training script + model registry
    ├── inference/             # test_images.py for predictions
    ├── visualizations.py
    ├── metrics.py
    └── __init__.py
```

---

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

Uses:

* Python 3.10+
* numpy
* scikit-learn
* pillow
* xgboost
* joblib

---

## Training a Model

Training is done through the CLI in `src/training/train_mnist.py`.

List available models:

```
python3 -m src.train_mnist --list-models
```

Train one model:

```
python3 -m src.train_mnist --model <model_name>
```

Example:

```
python3 -m src.train_mnist --model linear_svm
```

This loads MNIST, trains the model, prints accuracy, and saves:

```
models/<model_name>.joblib
```

## Evaluate a model
```bash 
python3 -m src.evaluate --model <model>
```

---
python -m src.evaluate --model linear_svm
## Testing on PNG Images

Put your PNGs in:

```
our-test-images/
```

Run:

```
python3 -m src.inference.test_images <model_name>
```

Example:

```
python3 -m src.inference.test_images linear_svm
```

The script loads the model, preprocesses each PNG into MNIST format, and prints predictions.
>>>>>>> fb7635263de5caa7df86942047429e4e1ed6fff5
