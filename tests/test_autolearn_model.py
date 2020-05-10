import pandas as pd
from sklearn.datasets import load_boston

import autolearn

boston = load_boston()
features = boston.feature_names
target = "MEDV"
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data[target] = boston.target


class TestModel:
    @classmethod
    def setup_class(cls):
        cls.x = data[features]
        cls.y = data[target]

    def test_tune(self):
        model = autolearn.Model(task="regression")
        model.tune(x=self.x, y=self.y, test_ratio=0.8, n_trials=10, n_jobs=-1)
        assert hasattr(model, "params")


class TestHyperparams:
    def test_get_hyperparams_ranges_regression(self):
        params = autolearn.Model._get_params_ranges("regression")
        assert isinstance(params, dict)
