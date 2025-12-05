import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path so src imports work if running directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.data_loading import load_mnist
from src.utils.paths import DATA_DIR
from src.utils.utils import load_saved_model

def parse_args():
    """Parse arguments to support two modes: 'evaluate' for plotting all results
    and 'compare' for inspecting individual predictions."""
    
    parser = argparse.ArgumentParser(description="Evaluate or compare robustness of SVM models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    eval_parser = subparsers.add_parser('evaluate', help='Run full robustness evaluation and plot.')
    
    eval_parser.add_argument(
        "--model", 
        type=str, 
        nargs='+', 
        required=True, 
        help="List of model names to evaluate (e.g. svm_linear svm_poly svm_rbf)"
    )
    eval_parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory where the .joblib model files are located."
    )
    eval_parser.add_argument(
        "--output_plot",
        type=str,
        default="robustness_comparison.png",
        help="Filename to save the comparison plot."
    )

    comp_parser = subparsers.add_parser('compare', help='Inspect individual predictions for a single noise level.')
    
    comp_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Single model name to compare (e.g. svm_rbf)"
    )
    comp_parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory where the .joblib model files are located."
    )
    comp_parser.add_argument(
        "--numbers", 
        type=int, 
        nargs='+', 
        required=True, 
        help="List of sample indices from the test set to display (e.g. 1 5 100)"
    )
    comp_parser.add_argument(
        "--noise-level", 
        type=float, 
        required=True, 
        help="Gaussian noise level (sigma) as a fraction of the data range [0, 1] (e.g. 0.3 for 30%% noise)"
    )

    return parser.parse_args()

def load_test_data():
    """
    Loads MNIST test data.
    NOTE: Assumes the data loading pipeline now returns data scaled to [0, 1].
    """
    print("Loading MNIST test data...")
    _, _, X_test, y_test = load_mnist(DATA_DIR) 
    return X_test, y_test

def add_gaussian_noise(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies Additive White Gaussian Noise (AWGN) to [0, 1] data.
    
    Args:
        X: Input data (N, D) in range [0, 1]
        sigma: Standard deviation of the noise distribution
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    # CHANGED: Clip to 1.0 because data is now normalized to [0, 1]
    return np.clip(X + noise, 0., 1.)

def evaluate_single_model(model_name, model_path, X_test, y_test, noise_levels):
    """
    Loads a model, evaluates it across noise levels, and prints an ASCII table.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*60}")

    try:
        model = load_saved_model(model_name, model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return []

    results = []
    
    for p in noise_levels:
        sigma = p 
        
        if sigma > 0:
            X_noisy = add_gaussian_noise(X_test.copy(), sigma)
        else:
            X_noisy = X_test
            
        acc = model.score(X_noisy, y_test)
        
        results.append({
            "Noise Pct": p,
            "Noise Label": f"{int(p*100)}%",
            "Sigma": sigma,
            "Accuracy": acc
        })
        
        print(f"  -> Noise: {int(p*100)}% (sigma={sigma:.2f}) | Accuracy: {acc:.4f}")

    # Print ASCII Table for this model
    df = pd.DataFrame(results)
    df_display = df.copy()
    # Format sigma to 2 decimal places since they are small floats now (0.10, 0.20...)
    df_display['Sigma'] = df_display['Sigma'].map('{:.2f}'.format)
    df_display['Accuracy'] = df_display['Accuracy'].map('{:.4f}'.format)
    
    print(f"\nSummary Table for {model_name}:")
    try:
        print(df_display[['Noise Label', 'Sigma', 'Accuracy']].to_markdown(index=False))
    except ImportError:
        print(df_display[['Noise Label', 'Sigma', 'Accuracy']].to_string(index=False))
        
    return results

def plot_comparison(all_results, output_filename):
    """Generates and saves a comparison plot."""
    if not all_results:
        print("No results to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # Define styles for known models, default for others
    markers = {'svm_linear': 'o', 'svm_poly': 's', 'svm_rbf': '^'}
    colors = {'svm_linear': 'blue', 'svm_poly': 'green', 'svm_rbf': 'red'}

    for model_name, data in all_results.items():
        if not data: continue
        
        df = pd.DataFrame(data)
        x_vals = df['Noise Pct'] * 100
        y_vals = df['Accuracy']
        
        plt.plot(x_vals, y_vals, 
                 marker=markers.get(model_name, 'o'), 
                 color=colors.get(model_name, None), 
                 linewidth=2, 
                 label=model_name)

    plt.title("SVM Robustness to Gaussian Noise", fontsize=14)
    plt.xlabel("Noise Level (%)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    print(f"\nSaving comparison plot to {output_filename}...")
    plt.savefig(output_filename)


def display_digit_with_noise(original_sample, noisy_sample, true_label, prediction, model_name, noise_pct):
    """
    Displays the original and noisy sample side-by-side using matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    # Original Image
    axes[0].imshow(original_sample.reshape(28, 28), cmap='gray')
    axes[0].set_title(f"Original: {true_label}", fontsize=10)
    axes[0].axis('off')

    # Noisy Image
    axes[1].imshow(noisy_sample.reshape(28, 28), cmap='gray')
    axes[1].set_title(f"Noisy ({noise_pct*100:.0f}%): {prediction}", fontsize=10)
    axes[1].axis('off')

    fig.suptitle(f"Model: {model_name} | True: {true_label} | Pred: {prediction}", fontsize=12)
    plt.show()

def print_prediction_comparison(model_name: str, model_path: Path, X_test: np.ndarray, y_test: np.ndarray, indices: list[int], noise_level: float):
    """
    Loads a model, applies noise, and displays a comparison for selected samples.
    """
    print(f"\n{'='*60}")
    print(f"PREDICTION COMPARISON FOR MODEL: {model_name}")
    print(f"Noise Level: {noise_level*100:.0f}% (sigma={noise_level:.2f})")
    print(f"Selected Sample Indices: {indices}")
    print(f"{'='*60}")

    try:
        model = load_saved_model(model_name, model_path)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return

    sigma = noise_level
    
    X_noisy = add_gaussian_noise(X_test.copy(), sigma) if sigma > 0 else X_test.copy()

    for idx in indices:
        if idx >= len(X_test):
            print(f"Warning: Index {idx} out of range for test set size {len(X_test)}. Skipping.")
            continue

        original_sample = X_test[idx]
        noisy_sample = X_noisy[idx]
        true_label = y_test[idx]
        
        prediction = model.predict(noisy_sample.reshape(1, -1))[0]

        is_correct = "CORRECT" if prediction == true_label else "INCORRECT"

        print(f"\n--- Sample Index {idx} ---")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {prediction} ({is_correct})")
        display_digit_with_noise(original_sample, noisy_sample, true_label, prediction, model_name, noise_level)

def main():
    args = parse_args()
    X_test, y_test = load_test_data()

    if args.command == 'evaluate':
        noise_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        all_model_results = {}

        for model_name in args.model:
            model_results = evaluate_single_model(
                model_name, 
                args.dir, 
                X_test, 
                y_test, 
                noise_percentages
            )
            if model_results:
                all_model_results[model_name] = model_results

        plot_comparison(all_model_results, args.output_plot)
        
    elif args.command == 'compare':
        print_prediction_comparison(
            model_name=args.model,
            model_path=args.dir,
            X_test=X_test,
            y_test=y_test,
            indices=args.numbers,
            noise_level=args.noise_level
        )

if __name__ == "__main__":
    main()
