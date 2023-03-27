import os
import sys
from numpy import ndarray
from dataclasses import dataclass
from scipy.sparse._csc import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from pickle import dump
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


@dataclass
class Params_rdmf:
    X_train: csc_matrix
    y_train: ndarray
    _n_estimators: int
    _max_depth: int


def evaluate_rdmf(config: Params_rdmf) -> float:
    """
    Train, fit and score the model
    to test data to evaluate the model
    """
    rdmf = RandomForestClassifier(
        max_depth=config._max_depth,
        n_estimators=config._n_estimators)
    rdmf.fit(config.X_train, config.y_train)
    dump(rdmf, open('./data/models/rdmf_model.pkl', 'wb'))
    return rdmf.score(config.X_train, config.y_train)
