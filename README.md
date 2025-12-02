# MNIST Classification Project

This project implements and compares various machine learning models on the MNIST dataset. It includes pipelines for training, evaluation, and inference on custom handwritten digits.

## Project Structure

```text
.
├── data/                   # MNIST dataset (ubyte files)
├── models/                 # Serialized trained models (.joblib)
├── our-test-images/        # Custom PNG images for inference testing
├── src/                    # Source code package
│   ├── inference/          # Scripts for running predictions on new data
│   ├── training/           # Model registry and training logic
│   ├── utils/              # Specific utilities (paths, data loading)
│   ├── metrics.py          # Evaluation metric functions
│   ├── test.py             # Script for testing models on custom PNG images
│   ├── train_mnist.py      # Main entry point for training
│   └── visualizations.py   # Plotting and visual analysis
└── requirements.txt        # Python dependencies
````

## Setup & Installation

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    Ensure you are using Python 3.10+.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

**Note:** All commands should be run from the project root directory using the `-m` flag to ensure Python handles relative imports within the `src` package correctly.

### 1\. Training Models

The training script requires you to specify which model you want to train.

**List available models:**

```bash
python -m src.train_mnist --list-models
```

**Train a specific model:**
This command trains the model on MNIST data, prints the accuracy/confusion matrix, and saves the serialized model to the `models/` directory.

```bash
python -m src.train_mnist --model linear_svm
# or
python -m src.train_mnist --model xgboost
```

### 2\. Inference on Custom Images

To test a trained model on the custom handwritten PNG digits located in `our-test-images/`, use the `src.test` module. You must provide the name of the model you wish to use (which must exist in `models/`).

```bash
# Syntax: python -m src.test <model_name>
python -m src.test linear_svm
```

### 3\. Visualizations

To run dimensionality reduction (PCA, t-SNE) and view random examples from the dataset:

```bash
python -m src.visualizations
```

## Data & Paths

  * **Dataset:** The project expects standard MNIST binary files (`.idx3-ubyte`) in `data/MNIST/`.
  * **Custom Images:** PNG images for inference should be placed in `our-test-images/`. They will be automatically resized to 28x28 and converted to grayscale.
  * **Path Management:** Absolute paths are handled dynamically via `src/utils/paths.py`.

## Results and Metrics

  * **Training Output:** Accuracy and Confusion Matrices are printed to the console immediately after training.
  * **Metric Analysis:** The `src.metrics` module contains helper functions like `summarize_confusions` to analyze specific error patterns (e.g., 4 vs 9), which are utilized during the training evaluation pipeline.


