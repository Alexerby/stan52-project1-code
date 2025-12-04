from sklearn.svm import SVC, LinearSVC

MODEL_REGISTRY = {
    "linear_svm": {
        "scaler": True,
        "model": LinearSVC(dual=False),
        "param_grid": {}
    },
    "svm_rbf": {
        "scaler": True,
        "model": SVC(kernel="rbf"),
        "param_grid": {
            "model__C": [0, 1, 10],
            "model__gamma": [0.001, 0.005]
        }
    },
}
