from pathlib import Path

from .train_mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def scatter_tsne_2d(X, y, n_samples=1000, perplexity=30):
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
    )
    X2 = tsne.fit_transform(X_sub)

    plt.figure(figsize=(7, 7))

    scatter = plt.scatter(
        X2[:, 0], X2[:, 1],
        c=y_sub,
        s=2,
        cmap="tab10"
    )

    # unique digits actually present in the sampled subset
    digits_present = np.unique(y_sub)

    # create legend handles only for digits present
    handles = [
        plt.Line2D(
            [], [], marker="o", color=plt.cm.tab10(d / 10), linestyle="", markersize=6
        )
        for d in digits_present
    ]

    plt.legend(
        handles,
        [str(d) for d in digits_present],
        title="Digit",
        loc="best",
        fontsize=8
    )

    plt.title(f"t-SNE 2D Projection ({n_samples} samples)")
    plt.axis("off")
    plt.show()


def show_random_examples(X, y, n=10):
    idx = np.random.choice(len(X), n, replace=False)
    plt.figure(figsize=(n, 2))
    for i, j in enumerate(idx):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[j].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()


def scatter_pca_2d(X, y):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=y, s=5, cmap="tab10")
    plt.title("PCA 2D Projection")
    plt.show()


if __name__ == "__main__":
    data_path = "data/MNIST"
    X_train, y_train, X_test, y_test = load_mnist(Path(data_path))

    show_random_examples(X_train, y_train)
    scatter_pca_2d(X_train, y_train)
    scatter_tsne_2d(X_train, y_train, n_samples=10_000)
