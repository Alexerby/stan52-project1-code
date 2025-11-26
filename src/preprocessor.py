"""
Utility functions for loading MNIST data from IDX file format.

The MNIST dataset is distributed as binary IDX files containing images and
labels. These helpers read the files directly and return NumPy arrays suitable
for use with scikit-learn models.

API: 
    load_mnist(data_dir)
"""

import numpy as np
from pathlib import Path


def _load_idx_images(path: Path) -> np.ndarray:
    """
    Load MNIST image data from an IDX-formatted file.

    Parameters
    ----------
    path : Path
        Filesystem path to the IDX image file.

    Returns
    -------
    np.ndarray
        A float32 array of shape (n_samples, n_rows * n_cols) containing
        flattened grayscale images.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Byte [0, 3] is just filetype, we don't need to store this ("magic number")
    n_images = int.from_bytes(data[4:8], byteorder="big")
    n_rows   = int.from_bytes(data[8:12], byteorder="big")
    n_cols   = int.from_bytes(data[12:16], byteorder="big")

    pixels = np.frombuffer(data, dtype=np.uint8, offset=16) # 1D here
    return pixels.reshape(n_images, n_rows * n_cols).astype(np.float32) # reshape to 2D


def _load_idx_labels(path: Path) -> np.ndarray:
    """
    Load MNIST label data from an IDX-formatted file.

    Parameters
    ----------
    path : Path
        Filesystem path to the IDX label file.

    Returns
    -------
    np.ndarray
        A uint8 array of shape (n_samples,) containing digit labels 0–9.
    """
    with open(path, "rb") as f:
        data = f.read()

    return np.frombuffer(data, dtype=np.uint8, offset=8)


def load_mnist(data_dir: Path):
    """
    Load MNIST from IDX files.
    """
    X_train = _load_idx_images(data_dir / "train-images.idx3-ubyte")
    y_train = _load_idx_labels(data_dir / "train-labels.idx1-ubyte")
    X_test  = _load_idx_images(data_dir / "t10k-images.idx3-ubyte")
    y_test  = _load_idx_labels(data_dir / "t10k-labels.idx1-ubyte")

    return X_train, y_train, X_test, y_test

