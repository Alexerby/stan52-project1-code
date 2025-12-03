
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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
            "model__C": [1, 10, 100], 
            "model__gamma": [0.001, 0.01, 0.1, "scale"]
        }
    },
}
