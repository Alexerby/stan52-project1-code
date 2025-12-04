import argparse
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from src.utils.utils import time_execution

from .utils.paths import DATA_DIR, MODEL_DIR
from .registry import MODEL_REGISTRY
from src.utils.data_loading import load_mnist


def make_pipeline(spec: dict) -> Pipeline:
    """
    Build a Pipeline from a model specification.
    """
    steps = []
    if spec["scaler"]:
        steps.append(("scaler", StandardScaler()))
    
    steps.append(("model", spec["model"])) 
    return Pipeline(steps, verbose=True)


def run_hyperparameter_tuning(model_name: str, spec: dict, X_train, y_train):
    """
    Performs K-Fold Cross-Validation to find the best hyperparameters.
    """
    print(f"Starting Grid Search for {model_name}...")
    print(f"Grid: {spec['param_grid']}")
    
    pipeline = make_pipeline(spec)
    
    grid_search = GridSearchCV(
        pipeline, 
        spec['param_grid'], 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=3
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n--- Best Parameters Found ---")
    print(grid_search.best_params_)
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_and_evaluate(model_name: str, spec: dict, 
                       X_train, y_train, X_test, y_test, 
                       tune=False):
    """
    Train (or tune) one model and print evaluation metrics.
    """
    
    if tune and spec.get("param_grid"):
        pipeline = run_hyperparameter_tuning(model_name, spec, X_train, y_train)
    else:
        print(f"Training model: {model_name} (No tuning)")
        pipeline = make_pipeline(spec)
        pipeline.fit(X_train, y_train)

    print(f"\nEvaluating on Test Set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, out_path)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print(f"Saved model to: {out_path}\n")


def parse_args():
    """All the args to be parsed."""
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

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning (GridSearch) before final evaluation."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="To test model on a subset of 1 000 observations."
    )

    return parser.parse_args()

@time_execution
def main():
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for name in MODEL_REGISTRY:
            print(" ", name)
        return

    if args.model is None:
        raise SystemExit("Error: --model is required unless using --list-models.")

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{args.model}'. "
            f"Run with --list-models to see available options."
        )

    print(f"Loading MNIST from {DATA_DIR}...")
    X_train, y_train, X_test, y_test = load_mnist(DATA_DIR, 10_000)
    
    if args.test:
        test_no = 1_000
        X_train, y_train, X_test, y_test = (
            X_train[:test_no],
            y_train[:test_no],
            X_test[:test_no],
            y_test[:test_no]
        )

    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

    # Train
    spec = MODEL_REGISTRY[args.model]
    train_and_evaluate(
        args.model, 
        spec, 
        X_train, y_train, X_test, y_test, 
        tune=args.tune
    )

if __name__ == "__main__":
    main()
