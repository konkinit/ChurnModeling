import os
import sys
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Any
from xgboost import XGBClassifier
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    model_report,
    _sample_weight
)
from src.configs import (
    models_inputs,
    rdmf_configs,
    xgb_configs
)


class EnsembleModel:
    def __init__(
            self,
            model_name,
            X_train,
            y_train,
            X_test,
            y_test
    ) -> None:
        self.model_name = model_name
        self.inputs = models_inputs(
            X_train,
            y_train,
            X_test,
            y_test
        )

    def fit_and_save(self) -> None:
        """Configure, fit and save the model
        """
        self._model_config()
        self.model.fit(
            self.inputs.X_train,
            self.inputs.y_train,
            _sample_weight(self.inputs.y_train)
        )
        dump(
            self.model,
            open(f'./data/models/{self.model_name}.pkl', 'wb')
        )

    def inference(self, cutoff: float) -> Tuple[Any]:
        """Outputs inference on the model

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


class _RandomForest(EnsembleModel):
    def __init__(
            self,
            X_train,
            y_train,
            X_test,
            y_test
    ) -> None:
        super().__init__("rdf", X_train, y_train, X_test, y_test)

    def _model_config(self) -> None:
        self.model = RandomForestClassifier(**dict(rdmf_configs()))


class _XGBClassifier(EnsembleModel):
    def __init__(
            self,
            X_train,
            y_train,
            X_test,
            y_test
    ) -> None:
        super().__init__("xgb", X_train, y_train, X_test, y_test)

    def _model_config(self) -> None:
        self.model = XGBClassifier(**dict(xgb_configs()()))
