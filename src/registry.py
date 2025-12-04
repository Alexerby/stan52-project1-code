from sklearn.svm import SVC

C = [0.5, 1, 5]

MODEL_REGISTRY = {
    "svm_linear": {
        "scaler": True,
        "model": SVC(kernel="linear"),
        "param_grid": {
            "model__C": C,
        }
    },
    "svm_rbf": {
        "scaler": True,
        "model": SVC(kernel="rbf"),
        "param_grid": {
            "model__C": C,
            "model__gamma": [0.01, 0.05, 0.1]
        }
    },
    "svm_poly": {
        "scaler": True,
        "model": SVC(kernel="poly"),
        "param_grid": {
            "model__C": C,
            "model__degree": [3, 4, 5],
            "model__gamma": 1
        }
    },
}
