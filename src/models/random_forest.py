import os
import sys
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Any
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    model_report,
    _sample_weight,
    Params_rdmf
)


def train_rdmf(config: Params_rdmf) -> None:
    """
    Train, fit and score the model
    to test data to evaluate the model
    """
    rdmf = RandomForestClassifier(
        max_depth=config._max_depth,
        n_estimators=config._n_estimators)
    rdmf.fit(
        config.X_train,
        config.y_train,
        _sample_weight(config.y_train))
    dump(rdmf, open('./data/models/rdmf_model.pkl', 'wb'))


def evaluate_rdmf(
        config: Params_rdmf,
        cutoff: float) -> Tuple[Any]:
    """Train, fit and score the model to test data to evaluate the model

    Args:
        config (Params_rdmf): dataclass containing the model parameters
        cutoff (float): cutoff for transforming probabilities to discrete
        values

    Returns:
        Tuple[Any]: model report
    """
    rdmf = load(open('./data/models/rdmf_model.pkl', 'rb'))
    return model_report(rdmf, config, cutoff)
