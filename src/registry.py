from sklearn.svm import SVC, LinearSVC

MODEL_REGISTRY = {
    "svm_linear": {
        "scaler": True,
        "model": LinearSVC(dual=False),
        "param_grid": {}
    },
    "svm_rbf": {
        "scaler": True,
        "model": SVC(kernel="rbf"),
        "param_grid": {
            "model__gamma": [0.01, 0.05]
        }
    },
    "svm_poly": {
        "scaler": True,
        "model": SVC(kernel="poly",),
        "param_grid": {
            "model__degree": [3, 4, 5],
            "model__gamma": [1, "scale"]
        }
    },
}
