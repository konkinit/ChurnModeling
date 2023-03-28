from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from .utils import model_report, Params_rdmf


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
        compute_sample_weight(
                    class_weight='balanced',
                    y=config.y_train))
    dump(rdmf, open('./data/models/rdmf_model.pkl', 'wb'))


def evaluate_rdmf(
        config: Params_rdmf,
        cutoff: float):
    """
    Train, fit and score the model
    to test data to evaluate the model
    """
    rdmf = load(open('./data/models/rdmf_model.pkl', 'rb'))
    return model_report(rdmf, config, cutoff)
