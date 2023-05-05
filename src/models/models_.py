import os
import sys
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Any
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    model_report,
    _sample_weight
)
from src.configs import (
    models_inputs,
    rdmf_configs
)


class RandomForest_:
    def __init__(
            self,
            model_name,
            X_train,
            y_train,
            X_test,
            y_test):
        self.model_name = model_name
        self.inputs = models_inputs(
            X_train,
            y_train,
            X_test,
            y_test
        )
        self.model_config = RandomForestClassifier(
            n_estimators=rdmf_configs().n_estimator,
            max_depth=rdmf_configs().max_depth
        )

    def fit_and_save(self) -> None:
        self.model_config.fit(
            self.inputs.X_train,
            self.inputs.y_train,
            _sample_weight(self.inputs.y_train)
        )
        dump(
            self.model_config,
            open(f'./data/models/{self.model_name}.pkl', 'wb')
        )

    def inference(self, cutoff: float) -> Tuple[Any]:
        """_summary_

        Args:
            cutoff (float): cutoff for transforming probabilities to discrete
        values

        Returns:
            Tuple[Any]: model report
        """
        model = load(open(f'./data/models/{self.model_name}.pkl', 'rb'))
        return model_report(
                    model,
                    self.inputs,
                    cutoff
                )
