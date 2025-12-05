from sklearn.svm import SVC

C = [0.01, 0.1]

MODEL_REGISTRY = {
    "svm_linear": {
        "scaler": False,
        "model": SVC(kernel="linear", C = 5),
        "param_grid": {
            "model__C": C,
        }
    },
    "svm_rbf": {
        "scaler": False,
        "model": SVC(kernel="rbf", C = 5, gamma = "scale"),
        "param_grid": {
            "model__C": C,
            "model__gamma": [0.0001]
        }
    },
    "svm_poly": {
        "scaler": False,
        "model": SVC(kernel="poly", C = 5, degree = 3),
        "param_grid": {
            "model__C": C,
            "model__degree": [3, 4, 5],
            "model__gamma": [1]
        }
    },
}
