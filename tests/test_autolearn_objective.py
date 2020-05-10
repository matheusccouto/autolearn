import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.datasets import load_boston

import autolearn
from autolearn import utils

boston = load_boston()
features = boston.feature_names
target = "MEDV"
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data[target] = boston.target


class TestInit:
    @classmethod
    def setup_class(cls):
        cls.x = data[features]
        cls.y = data[target]
        cls.model = xgb.XGBRegressor
        cls.hyperparams = {"n_estimators": 1}
        cls.scorer = sklearn.metrics.make_scorer(
            sklearn.metrics.mean_squared_error,
            greater_is_better=False,
            squared=False,
        )

    def test_init_cv(self):
        objective = autolearn._Objective(
            x=self.x,
            y=self.y,
            model=self.model,
            params=self.hyperparams,
            scorer=self.scorer,
            cv=2,
            n_jobs=1,
        )
        return_ = objective(None)
        assert isinstance(return_, float)

    def test_init_cv_time_series(self):
        objective = autolearn._Objective(
            x=self.x,
            y=self.y,
            model=self.model,
            params=self.hyperparams,
            scorer=self.scorer,
            cv=2,
            time_series=True,
            n_jobs=1,
        )
        return_ = objective(None)
        assert isinstance(return_, float)

    def test_init_split(self):
        objective = autolearn._Objective(
            x=self.x,
            y=self.y,
            model=self.model,
            params=self.hyperparams,
            scorer=self.scorer,
            cv=None,
            test_ratio=0.2,
            n_jobs=1,
        )
        return_ = objective(None)
        assert isinstance(return_, float)

    def test_init_split_time_series(self):
        objective = autolearn._Objective(
            x=self.x,
            y=self.y,
            model=self.model,
            params=self.hyperparams,
            scorer=self.scorer,
            cv=None,
            test_ratio=0.2,
            time_series=True,
            n_jobs=1,
        )
        return_ = objective(None)
        assert isinstance(return_, float)


class TestHyperparams:
    def test_prepare_params(self):
        study = optuna.study.create_study()
        study.enqueue_trial({})
        trial = optuna.Trial(study, 0)

        params_file = r"../autolearn\params\xgboost.yml"
        params = utils.read_yaml(params_file)

        params = autolearn._Objective._eval_params(trial, params)
        assert isinstance(params, dict)
