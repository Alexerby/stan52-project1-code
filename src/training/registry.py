import numpy as np
from sklearn.svm import SVC, LinearSVC

MODEL_REGISTRY = {
    "linear_svm": {
        "scaler": False,
        "model": LinearSVC(dual=False),
        "param_grid": {}
    },
    "svm_rbf": {
        "scaler": True,
        "model": SVC(kernel="rbf"),
        "param_grid": {
            "model__C": np.logspace(0, 2, num=3),       # 1, 10, 100
            "model__gamma": np.logspace(-2, 0, num=3)   # 0.01, 0.1, 1
        }
    },
}
