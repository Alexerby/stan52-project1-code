from sklearn.svm import SVC

C = [0.1, 0.5, 1, 5]

MODEL_REGISTRY = {
    "svm_linear": {
        "scaler": False,

        "param_grid": {
            "model__C": C,
        }
    },
    "svm_rbf": {
        "scaler": False,
        "model": SVC(kernel="rbf"),
        "param_grid": {
            "model__C": C,
            "model__gamma": [0.0001, 0.001, 0.01]
        }
    },
    "svm_poly": {
        "scaler": False,
        "model": SVC(kernel="poly"),
        "param_grid": {
            "model__C": C,
            "model__degree": [3, 4, 5],
            "model__gamma": [1] # force to equal one
        }
    },
}
