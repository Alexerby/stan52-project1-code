## MNIST Classification Project

This project trains and evaluates machine learning models (including tuned SVMs) on the standard MNIST dataset and custom hand-drawn images.

-----

## Project Structure

| Directory | Purpose |
| :--- | :--- |
| `data/` | Contains the raw MNIST dataset files. |
| `models/` | Stores trained models as `.joblib` files. |
| `our-test-images/` | **Place your custom PNG digit images here for testing.** |
| `src/` | All Python source code (training, inference, utilities). |

-----

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

-----

## Workflow & Usage

The main scripts are executable via the command line.

### 1\. Training a Model

Training is handled by `src.train`. The available model list is defined in the `src/training/registry.py` file.

| Action | Command |
| :--- | :--- |
| **List Models** | `python3 -m src.train --list-models` |
| **Train Standard** | `python3 -m src.train --model <model_name>` |
| **Tune & Train** | `python3 -m src.train --model <model_name> --tune` |

#### Tuning (`--tune`)

The `--tune` flag executes a **Grid Search Cross-Validation** (CV) to find optimal hyperparameters (e.g., $C$ and $\gamma$ for SVM) on the training set.


#### Testing (`--test`)
This will only train the model on 1,000 observations. Used for testing.

### 2\. Evaluating a Model

Evaluate a previously trained model against the official MNIST test set (to get the confusion matrix and final accuracy):

```bash
python3 -m src.evaluate --model <model_name>
```

### 3\. Testing on Custom Images

To see how a model performs on your own handwritten digits:

1.  Place your PNG images (e.g., `my_digit_4.png`) inside the `our-test-images/` directory.
2.  Run the inference script:

<!-- end list -->

```bash
python3 -m src.inference.test_images <model_name>
```

