
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
}
