# MNIST Classification Project

## 📂 Project Structure

| Directory | Purpose |
| :--- | :--- |
| `data/` | Contains the raw MNIST dataset files. |
| `models/` | Stores trained models (e.g., `svm_rbf.joblib`). |
| `our-test-images/` | **Place your custom PNG digit images here for testing.** |
| `src/` | Source code (`train.py`, `evaluate.py`, `test_custom_images.py`, etc.). |

---

## ⚙️ Setup

Install required dependencies:

```bash
pip install -r requirements.txt
````


**Requirements:** 
- Python >= 3.11


-----

## 🚀 Usage Workflow

### Training (`src.train`)

Training is handled by `src.train`. The available model list is defined in `src/registry.py`.

**Common Commands:**

```bash
# List all available models
python -m src.train --list-models

# Train a specific model (e.g., SVM)
python -m src.train --model svm_rbf

# Train with Hyperparameter Tuning (Grid Search)
python -m src.train --model svm_rbf --tune
```

**Configuration Flags:**

| Flag | Description |
| :--- | :--- |
| `--model <name>` | **Target Model:** Specifies the model architecture to train (e.g., `svm_rbf`). |
| `--list-models` | **Info:** Lists all available model names in the registry and exits. |
| `--tune` | **Optimization:** Executes **Grid Search Cross-Validation** to find optimal hyperparameters (e.g., $C$ and $\gamma$) before final training. |
| `--test` | **Debug Mode:** Trains on a subset of **1,000 observations** only. Use this to verify the pipeline works without waiting for full training. |
| `--full` | Train the model on the entire MNIST dataset |

###  Evaluation (`src.evaluate`)

Evaluate a previously trained model against the official MNIST test set to generate a confusion matrix and accuracy score:

```bash
python -m src.evaluate --model svm_rbf
```
| Flag | Description |
| :--- | :--- |
| `--dir` | **Debug Mode:** Specify path to another dir where the model exists. |



###  Inference (`src.test_custom_images`)

To see how a model performs on your own handwritten digits:

1.  Place your PNG images (e.g., `my_digit_4.png`) inside the `our-test-images/` directory.
2.  Run the custom image test script.

**Basic Usage:**

```bash
python -m src.test_custom_images svm_rbf
```

**Arguments:**

| Argument | Type | Description |
| :--- | :--- | :--- |
| `model` | **Positional** | The name of the saved model (without `.joblib` extension). |
| `--dir` | **Optional** | Directory containing PNG images. Defaults to `our-test-images/`. |

```
```


###  Visualizations (`src.visualizations`)
**Basic Usage:**

```bash
python -m src.visualizations
```


### Robustness Testing and Visualization (`src.corruption`)

This script is used to evaluate model robustness by applying **Additive White Gaussian Noise (AWGN)** to the test data and analyzing the resultant drop in accuracy. It supports two modes:

#### Robustness Evaluation (`evaluate`)

Runs a full robustness test for one or more models across a range of noise levels (0% to 100%) and generates a comparison plot, saving it as a PNG file.

**Basic Usage:**

```bash
python -m src.corruption evaluate \
    --model svm_linear svm_poly svm_rbf \
    --dir models/saved/ \
    --output_plot robustness_comparison_svms.png
```

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--model <names>` | **Required** | List of model names (separated by spaces) to compare. |
| `--dir <path>` | **Required** | Directory where the `.joblib` model files are located. |
| `--output_plot <filename>`| **Optional** | Filename to save the resulting accuracy-vs-noise plot. |

-----

#### Prediction Comparison (`compare`)

Inspects individual predictions for a single model at a specified noise level. It displays the original digit and the corrupted digit side-by-side using Matplotlib, along with the true label and the model's prediction.

**Basic Usage (example 30% noise, numbers 1, 5, 9):**

```bash
python -m src.corruption compare \
    --model svm_rbf \
    --dir /abs/path/to/dir/ \
    --numbers 1 5 9 \
    --noise-level 0.3
```

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--model <name>` | **Required** | The single model name to test. |
| `--dir <path>` | **Required** | Directory where the `.joblib` model file is located. |
| `--numbers <indices>`| **Required** | List of **sample indices** from the MNIST test set to display. |
| `--noise-level <sigma>` | **Required** | The standard deviation ($\sigma$) of the Gaussian noise, as a float in the range $[0.0, 1.0]$. (e.g., $0.3$ for $30\%$ noise). |
