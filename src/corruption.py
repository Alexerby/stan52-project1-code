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
    """Parse arguments to accept multiple models."""
    parser = argparse.ArgumentParser(description="Evaluate robustness of specific SVM models.")
    
    # CHANGED: nargs='+' allows multiple arguments like: --model svm_linear svm_poly
    parser.add_argument(
        "--model", 
        type=str, 
        nargs='+', 
        required=True, 
        help="List of model names to evaluate (e.g. svm_linear svm_poly svm_rbf)"
    )
    
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory where the .joblib model files are located."
    )
    
    parser.add_argument(
        "--output_plot",
        type=str,
        default="robustness_comparison.png",
        help="Filename to save the comparison plot."
    )

    return parser.parse_args()

def load_test_data():
    print("Loading MNIST test data...")
    _, _, X_test, y_test = load_mnist(DATA_DIR) 
    return X_test, y_test

def add_gaussian_noise(X: np.ndarray, sigma: float) -> np.ndarray:
    """Applies Additive White Gaussian Noise (AWGN) to [0, 255] data."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return np.clip(X + noise, 0., 255.)

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
        sigma = p * 255.0
        
        # Apply noise (using .copy() to preserve original data)
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
        
        print(f"  -> Noise: {int(p*100)}% (sigma={sigma:.1f}) | Accuracy: {acc:.4f}")

    # Print ASCII Table for this model
    df = pd.DataFrame(results)
    df_display = df.copy()
    df_display['Sigma'] = df_display['Sigma'].map('{:.1f}'.format)
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
                 color=colors.get(model_name, None), # None lets matplotlib pick a color if unknown
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
    # plt.show() # Uncomment if you want the window to pop up

def main():
    args = parse_args()
    X_test, y_test = load_test_data()

    # Define noise levels (0% to 90%)
    noise_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    all_model_results = {}

    # Iterate over the models provided in the command line argument
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

    # Plot Results
    plot_comparison(all_model_results, args.output_plot)

if __name__ == "__main__":
    main()
