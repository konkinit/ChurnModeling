import xgboost
from dataclasses import dataclass
from numpy import ndarray
from scipy.sparse._csc import csc_matrix


@dataclass
class Params_xgb:
    X_train: csc_matrix
    y_train: ndarray
    _n_estimators: int
    _max_depth: int


def evaluate_xgboost(config: Params_xgb):
    pass
