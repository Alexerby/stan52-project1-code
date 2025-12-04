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
