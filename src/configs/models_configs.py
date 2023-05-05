from dataclasses import dataclass
from ml_collections import ConfigDict
from numpy import ndarray
from scipy.sparse import csc_matrix


def rdmf_configs() -> ConfigDict:
    cfg = ConfigDict()
    cfg.max_depth = 2
    cfg.n_estimators = 500
    return cfg


def xgb_configs():
    cfg = ConfigDict()
    cfg.max_depth = 2
    cfg.n_estimators = 500
    return cfg


@dataclass
class models_inputs:
    X_train: csc_matrix
    y_train: ndarray
    X_test: csc_matrix
    y_test: ndarray
