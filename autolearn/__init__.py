""" Automated Machine Learning. """

import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb

from autolearn import utils


def split(
    x: pd.DataFrame,
    y: pd.Series,
    test_samples: Optional[int] = None,
    test_ratio: Optional[float] = None,
    time_series: bool = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split  dataset into a train and test params.

    If neither are passed, 0.2 ratio is used. When both are passed, uses
    min(test_samples, test_ratio).

    Args:
        x: Features dataframe.
        y: Target series.
        test_samples: Size of the test set.
        test_ratio: Ratio size of the test size.
        time_series: Use time series splitting methodology.
        random_state: Seed used by random number generator.

    Returns:
        x train, x test, y train, y test
    """
    default_ratio = 0.2

    size = x.shape[0]

    # When neither are passed use the default value.
    if not test_samples and not test_ratio:
        test_ratio = default_ratio

    if not test_ratio:
        test_ratio = np.nan

    if not test_samples:
        test_samples = np.nan

    test_sizes = [np.rint(test_ratio * size), test_samples]
    test_size = int(np.nanmin(test_sizes))

    if time_series:
        train_size = size - test_size
        return x[:train_size], x[train_size:], y[:train_size], y[train_size:]

    return sklearn.model_selection.train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )


class _Objective:
    """
    Optuna objective.

    The purpose of this class is to create an object with arguments
    other than trial.
    """

    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        model: Callable,
        params: Dict[str, Any],
        scorer: Callable,
        cv: Optional[int] = 5,
        test_samples: Optional[int] = None,
        test_ratio: Optional[float] = None,
        time_series: bool = False,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        """
        Args:
            x: Features dataframe.
            y: Target dataframe.
            model: Estimator.
            params: Estimator hyperparameters.
            scorer: Scorer function.
            cv: Number of folds. None to not use cross validation.
            test_samples: Max samples in the test set. Only valid when
                not using cross validation.
            test_ratio: Max ratio of the test set. Only valid when not
                using cross validation.
            time_series: Use time series splitting methodology.
            random_state: Seed used by random number generator.
            n_jobs: Number of CPUs to use. -1 to use all available.
        """
        self._x = x
        self._y = y
        self._model = model
        self._params = params
        self._scorer = scorer
        if test_ratio or test_samples:
            cv = None
        self._cv = cv
        self._test_samples = test_samples
        self._test_ratio = test_ratio
        self._time_series = time_series
        self._random_state = random_state
        self._n_jobs = n_jobs

    @staticmethod
    def _cross_validate(
        x: pd.DataFrame,
        y: pd.Series,
        model: Callable,
        scorer: Callable,
        cv: int = 5,
        time_series: bool = False,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ) -> float:
        """
        Score a estimator using cross validate.

        Args:
            x: Features dataframe.
            y: Target series.
            model: Estimator instance.
            scorer: Scorer function.
            cv: Number of folds.
            time_series: Use time series splitting methodology.
            random_state: Seed used by random number generator.
            n_jobs: Number of CPUs to use. -1 to use all available.

        Returns:
            Score.
        """
        if time_series:
            k_fold = sklearn.model_selection.TimeSeriesSplit(n_splits=cv)
        else:
            k_fold = sklearn.model_selection.KFold(
                n_splits=cv, shuffle=False, random_state=random_state
            )

        return sklearn.model_selection.cross_val_score(
            model, x, y, scoring=scorer, cv=k_fold, n_jobs=n_jobs,
        ).mean()

    @staticmethod
    def _train_test(
        x: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBModel,
        scorer: Callable,
        test_samples: Optional[int] = None,
        test_ratio: Optional[float] = None,
        time_series: bool = False,
        random_state: Optional[int] = None,
    ):
        datasets = split(
            x, y, test_samples, test_ratio, time_series, random_state
        )
        x_train, x_test, y_train, y_test = datasets

        model.fit(x_train, y_train)
        return scorer(model, x_test, y_test)

    @staticmethod
    def _eval_params(trial, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate args for optuna.

        Args:
            trial: optuna arg.
            params: Model hyperparams.

        Returns:
            Kwargs dict
        """
        prepared = dict()
        for arg, value in params.items():
            if isinstance(value, dict):
                # Extract method.
                name = list(value.keys())[0]
                # Add prefix.
                method = "suggest_" + name
                # Get method kwargs.
                kwargs = value[name]
                # Add name arg.
                kwargs.update({"name": arg})
                # Evaluate method.
                value = getattr(trial, method)(**kwargs)
            prepared.update({arg: value})
        return prepared

    def __call__(self, trial) -> float:
        """
        Method that will be called by optuna.

        Args:
            trial: optuna arg.

        Returns:
            Trial score.
        """
        params = self._eval_params(trial, self._params)
        model = self._model(**params)

        if self._cv:
            return self._cross_validate(
                x=self._x,
                y=self._y,
                model=model,
                scorer=self._scorer,
                cv=self._cv,
                time_series=self._time_series,
                random_state=self._random_state,
                n_jobs=self._n_jobs,
            )
        else:
            return self._train_test(
                x=self._x,
                y=self._y,
                model=model,
                scorer=self._scorer,
                test_samples=self._test_samples,
                test_ratio=self._test_ratio,
                time_series=self._time_series,
                random_state=self._random_state,
            )


class Model:
    """ Machine learning model. Uses XGBModel. """

    def __init__(self, task: str, verbose: bool = True):
        """
        Args:
            task: Learning task.
                * "regression"
                * "time series regression"
                * "binary classification"
                * "time series binary classification
            verbose: Verbosity.
        """
        self._task = task
        self._estimator = xgb.XGBModel
        self._time_series = "time series" in task

        self._params = {}
        self._model = self._estimator()
        self._scorer = self._get_scorer(self._task)

        if verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

    @property
    def task(self) -> str:
        """ Get or set learning task. """
        return self._task

    @task.setter
    def task(self, value: str):
        self._task = self._validate_task(value)
        self._scorer = self._get_scorer(self._task)

    @property
    def time_series(self) -> bool:
        """ Get or set if learning task is a time series. """
        return self._time_series

    @time_series.setter
    def time_series(self, value: bool):
        self._time_series = value

    @property
    def params(self) -> Dict[str, Any]:
        """ Get or set XGBModel parameters. """
        if not self._params:
            raise AttributeError("Parameters are not available yet.")
        return self._params

    # TODO Create a params validation method.
    # TODO Make it possible to read params from a YAML file.
    @params.setter
    def params(self, value: Dict[str, Any]):
        self._params = value

    @staticmethod
    def _validate_task(task: str) -> str:
        """
        Validate learning task argument.

        Args:
            task: Learning task.

        Returns:
            Learning task.

        Raises:
            ValuerError: When learning task is not supported.
        """
        supported = ("regression", "binary")
        if not any([task in item for item in supported]):
            raise ValueError(f"{task} is not a supported task.")
        return task

    @staticmethod
    def _get_scorer(task: str) -> Callable:
        """
        Get model scorer.

        Args:
            task: Learning task.

        Returns:
            Callable object that returns a scalar score. Less is better.

        Raises:
            ValuerError: When learning task is not supported.
        """
        if "regression" in task.lower():
            return utils.negate(
                sklearn.metrics.make_scorer(
                    sklearn.metrics.mean_squared_error, squared=False
                )
            )
        if "binary" in task.lower():
            return sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
        raise ValueError(f"{task} is not a supported task.")

    @staticmethod
    def _get_params_ranges(task: str,) -> Dict[str, Any]:
        """
        Get suitable hyperparameters ranges for a given task.

        The returning dictionary may contain other dictionaries in the
        values of a argument. This indicates that this is a optuna.trial
        method. The class _Objective will handle transforming
        this dictionary into a callable.

        Args:
            task: Learning task.

        Returns:
            Hyperparameters dictionary.

        Raises:
            ValuerError: When learning task is not supported.
        """
        params_file = os.path.join(
            os.path.dirname(__file__), "params", "xgboost.yml"
        )
        params = utils.read_yaml(params_file)

        if "regression" in task.lower():
            params.update({"objective": "reg:squarederror"})
            return params
        if "binary" in task.lower():
            params.update({"objective": "binary:logistic"})
            return params
        raise ValueError(f"{task} is not a supported task.")

    # TODO Optimize tuning doing in two steps.
    #  1 - With a smaller sample get param importance.
    #  2 - Only tune params with importance above a threshold.
    def tune(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        cv: Optional[int] = 5,
        test_samples: Optional[int] = None,
        test_ratio: Optional[float] = None,
        n_trials: int = 100,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        """
        Tune hyperparameters.

        Args:
            x: Features dataframe.
            y: Target series.
            cv: Number of folds. None to not use cross validation.
            test_samples: Max samples in the test set. Only valid when
                not using cross validation.
            test_ratio: Max ratio of the test set. Only valid when not
                using cross validation.
            n_trials: Number of trials for optimization algorithm.
            random_state: Seed used by random number generator.
            n_jobs: Number of CPUs to use. -1 to use all available.

        Returns:
            Dict with best hyperparameters.
        """
        params = self._get_params_ranges(self._task)
        # For tuning we'll use tree_method="hist" because it is faster.
        params.update({"tree_method": "hist"})

        objective = _Objective(
            x=x,
            y=y,
            model=self._estimator,
            params=params,
            scorer=self._scorer,
            cv=cv,
            test_samples=test_samples,
            test_ratio=test_ratio,
            time_series=self._time_series,
            random_state=random_state,
            n_jobs=1,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        self._params = study.best_params
        self._model = self._estimator(**self._params)

    def set_params(self):
        raise NotImplementedError

    def fit(self, x: pd.DataFrame, y: pd.Series, *args, **kwargs):
        self._model.fit(x, y, *args, **kwargs)

    def score(self, x: pd.DataFrame, y: pd.Series, *args, **kwargs):
        return self._scorer(self._model, x, y, *args, **kwargs)
