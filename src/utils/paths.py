"""
Centralized project paths.

This module provides absolute paths to the project root and its
subdirectories, independent of where Python is executed from.
"""

from pathlib import Path

# Directory containing this file: project/src/utils/
THIS_DIR = Path(__file__).resolve().parent

# Project root = two levels above utils/
PROJECT_ROOT = THIS_DIR.parent.parent

# Common paths
DATA_DIR = PROJECT_ROOT / "data" / "MNIST"
MODEL_DIR = PROJECT_ROOT / "models"
TEST_IMAGE_DIR = PROJECT_ROOT / "our-test-images"


def ensure_directories():
    """
    Create directories that must exist: models/, etc.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
