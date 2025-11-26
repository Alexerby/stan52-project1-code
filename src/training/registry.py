
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "linear_svm": {
        "scaler": False,
        "model": LinearSVC(dual=False)
    },
    "svm_rbf": {
        "scaler": True,
        "model": SVC(kernel="rbf", gamma="scale")
    },
    "svm_poly": {
        "scaler": True,
        "model": SVC(kernel="poly", degree=3)
    },
    "random_forest": {
        "scaler": False,
        "model": RandomForestClassifier(n_estimators=200)
    },
    "gradient_boosting": {
        "scaler": False,
        "model": GradientBoostingClassifier()
    },
    "xgboost": {
        "scaler": False,
        "model": XGBClassifier(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            subsample=0.8,
        )
    },
}
