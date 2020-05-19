""" Automated feature engineering and selection. """
# TODO: The ignore argument is used in several method.
#  There should be a function to avoid repeating this routine.

from typing import Union, Dict, Sequence, Optional, Tuple

import featuretools as ft
import numpy as np
import pandas as pd
import sklearn
from featuretools import selection

import autolearn
import rfpimp
from autolearn import utils


def load(
    entity_set_folder_name: str, features_file_name: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Load dataframe from featuretools params.
    
    Args:
        entity_set_folder_name: Entity set folder path.
        features_file_name: Features file path.
        verbose: Verbosity,

    Returns:
        Dataframe.
    """
    es = ft.read_entityset(entity_set_folder_name)
    features = ft.load_features(features_file_name)
    return ft.calculate_feature_matrix(features, es, verbose=verbose)


class _Operator:
    """ Base class for feature operation. """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        features: Optional[Sequence[str]] = None,
        task: str = "regression",
    ):
        """
        Args:
            data: DataFrame (features + target).
            target: Target feature name.
            features: Features name sequence.
            task: Learning task. "regression" or "classification"
        """
        if features is None:
            features = data.columns
        self._features = [feat for feat in features if feat not in target]
        self._target = target

        self._x = data.drop(self._target, axis=1)
        self._y = data[self._target]

        self._task = task

    @property
    def data(self) -> pd.DataFrame:
        """ Get or set data (features + target). """
        return self._x.join(self._y)

    @data.setter
    def data(self, value: pd.DataFrame):
        self._x = value.drop(self._target, axis=1)
        self._y = value[self._target]

    @property
    def features(self) -> Sequence:
        """ Get or set features names. """
        return self._features

    @features.setter
    def features(self, value: Sequence):
        self._features = [feat for feat in value if feat not in self._target]

    @property
    def target(self) -> str:
        """ Get or set target name. """
        return self._target

    @target.setter
    def target(self, value: str):
        self._target = value

    @property
    def task(self) -> str:
        """ Get or set learning task. """
        return self._task

    @task.setter
    def task(self, value: str):
        self._task = value


class _Evaluator(_Operator):
    """ Base class for feature evaluation. """

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        features: Optional[Sequence[str]] = None,
        task: str = "regression",
    ):
        super().__init__(data=data, target=target, features=features, task=task)
        self._importances = None
        self._dependence = None

    @property
    def importance(self):
        """ Get features importance dataframe. """
        if self._importances is None:
            raise AttributeError("Features importance are not yet available.")
        return self._importances

    @property
    def dependence(self):
        """ Get features dependence dataframe. """
        if self._dependence is None:
            raise AttributeError("Features dependence are not yet available.")
        return self._dependence

    @staticmethod
    def _manage_groups(
        groups: Dict[str, Sequence[str]], features: Sequence[str]
    ) -> Sequence[Sequence[str]]:
        """
        Transforms a dict of related features into a nested list for the
        the rfpimp.importance method.

        Args:
            groups: Dictionary of related features.
            features: All features to be considered.

        Returns:
            Nested list.
        """
        # Transforms the dictionary in a list of lists. That is the
        # format for the rfpimp.importance.
        nested = [values for values in groups.values()]

        # Flat the nested list and check which features were forgotten.
        flat = list(utils.flatten(nested))
        forgotten = [[feat] for feat in features if feat not in flat]

        # Return a feature list containing
        return nested + forgotten

    @staticmethod
    def _add_random_feature(
        data: pd.DataFrame, amount: int = 1
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """
        Add a feature of random values to a dataframe.

        Args:
            data: DataFrame
            amount: Amount of random features to add.

        Returns:
            Tuple with DataFrame and name of the random column.
        """
        names = list()
        random_feat_name = "random"
        for _ in range(amount):
            while random_feat_name in data.columns:
                random_feat_name += "_"
            data[random_feat_name] = np.random.random(size=data.shape[0])
            names.append(random_feat_name)
        return data, names

    # TODO Add CV option.
    def eval_importance(
        self,
        groups: Optional[Dict[str, Sequence[str]]] = None,
        n_times: int = 10,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Evaluate permutation feature importances.

        Args:
            groups: Groups of related features. One feature can appear
                on several groups at the same time.
            n_times: Number of times to calculate importances. Uses the
                mean of results.
            ignore: Features to ignore during dropping.
            n_jobs: Number of CPUs to use. -1 to use all available.

        Returns:
            DataFrame
        """
        # Prepare features list (os nested list).
        features = self._features
        if ignore:
            features = [feat for feat in features if feat not in ignore]
        if groups:
            features = self._manage_groups(groups, features)

        # Split dataset.
        n_samples = 5000
        ratio = 0.2
        datasets = autolearn.split(
            x=self._x, y=self._y, test_samples=n_samples, test_ratio=ratio
        )
        x_train, x_test, y_train, y_test = datasets

        model = autolearn.Model(task=self.task)
        model.tune(x_train, y_train, test_ratio=ratio, n_jobs=n_jobs)
        model.fit(x_train, y_train)

        kwargs = {
            "model": model,
            "X_valid": x_test,
            "y_valid": y_test,
            "features": features,
            "n_samples": -1,
        }

        # Get importances.
        imps = [rfpimp.importances(**kwargs) for _ in range(n_times)]
        imp = pd.concat(imps).groupby(level=0).mean()
        imp = imp.sort_values("Importance", ascending=False)

        # Create new columns.
        # Handle Negative values by adding its module to all values.
        non_negatives = imp["Importance"].add(np.abs(imp["Importance"].min()))
        imp["Normalised Importance"] = non_negatives / non_negatives.sum()
        imp["Cumulative Importance"] = imp["Normalised Importance"].cumsum()

        self._importances = imp
        return imp

    # TODO Add CV option.
    def eval_dependence(
        self, ignore: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Evaluate feature dependence matrix.

        Identify if a feature is dependent on other features.

        Args:
            ignore: Features to ignore during dropping.

        Returns:
            Feature dependence dataframe.
        """
        feats = self._features
        if ignore:
            feats = [feat for feat in self._features if feat not in ignore]
        x = self._x[feats]

        model = sklearn.ensemble.RandomForestRegressor(10, oob_score=True)
        return rfpimp.feature_dependence_matrix(x, rfmodel=model)


# TODO Add support to Agg Primitives
class Creator(_Operator):
    """ Automated feature creation. """

    _TRANS_PRIMITIVES: Sequence[str] = [
        "add_numeric",
        "subtract_numeric",
        "multiply_numeric",
        "divide_numeric",
        "greater_than",
        "less_than",
        "and",
        "or",
    ]

    @staticmethod
    def _set_group(features: Sequence[str]) -> Dict[str, Sequence[str]]:
        """
        Create a group with all features.

        Args:
            features: Features.

        Returns:
            Dictionary with a single key 'whole' and features as values.
        """
        return {"whole": features}

    @staticmethod
    def _get_extra_feats(
        features: Sequence[str], groups: Dict[str, Sequence[str]]
    ) -> Sequence[str]:
        """
        Get features in the groups that does not exist in the params.

        Args:
            features: Features.
            groups: Dict with features groups.

        Returns:
            Sequence of extra features.
        """
        group_features = [
            value for values in groups.values() for value in values
        ]
        return [
            group_feature
            for group_feature in group_features
            if group_feature not in features
        ]

    @staticmethod
    def _get_missing_feats(
        features: Sequence[str], groups: Dict[str, Sequence[str]]
    ) -> Sequence[str]:
        """
        Get missing features in the groups.

        Args:
            features: Features.
            groups: Dict with features groups.

        Returns:
            Sequence of groupless features.
        """
        group_features = [
            value for values in groups.values() for value in values
        ]
        return [
            feature for feature in features if feature not in group_features
        ]

    def _fix_groups(
        self,
        features: Sequence[str],
        groups: Dict[str, Sequence[str]],
        use_forgotten: bool = False,
    ) -> Dict[str, Sequence[str]]:
        """
        Fix groups of features.

        Args:
            features: Features.
            groups: Dict with features groups.
            use_forgotten:

        Returns:
            Features groups.
        """
        extra_features = self._get_extra_feats(features=features, groups=groups)
        if extra_features:
            raise ValueError(
                f"These features do not exist in the dataset: "
                f"{', '.join(extra_features)}."
            )

        missing_features = self._get_missing_feats(
            features=features, groups=groups
        )
        if missing_features and use_forgotten:
            groups.update({"forgotten": missing_features})

        return groups

    @staticmethod
    def _index_name(data: pd.DataFrame) -> str:
        """
        Get dataframe index name.

        Args:
            data: DataFrame.

        Returns:
            Index name.
        """
        index_name = data.index.name
        if index_name is None:
            # Default name is "index".
            index_name = "index"
        return index_name

    def _set_entity_set(
        self, data: pd.DataFrame, groups: Dict[str, Sequence[str]]
    ) -> ft.EntitySet:
        """
        Set an entity set.

        Args:
            data: DataFrame
            groups: Dict of features groups.

        Returns:
            Featuretools entity set.
        """
        es = ft.EntitySet(id="main")

        index_name = self._index_name(data)

        for group, features in groups.items():
            es = es.entity_from_dataframe(
                entity_id=group,
                dataframe=data[features].reset_index(),
                index=index_name,
            )

        return es

    def transform(
        self,
        groups: Optional[Dict[str, Sequence[str]]] = None,
        use_forgotten: bool = False,
        trans_primitives: Optional[Sequence[str]] = None,
        max_depth: int = 1,
        entity_set_folder_name: Optional[str] = None,
        features_file_name: Optional[str] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Create new features.

        Wraps Featuretools Deep Feature Synthesis.
        Default Featuretools trans primitives are:
            - "add_numeric"
            - "subtract_numeric"
            - "multiply_numeric"
            - "divide_numeric"
            - "greater_than"
            - "less_than"
            - "and"
            - "or"

        Use relationship groups to relate variables. This avoid wasting
        time creating features from other totally unrelated features.

        This is specially useful when working with datasets with several
        features. Be careful with bias.

        This method does not support multiples entities (consequently
        agg_primitives) yet. Groups are not entities, but only clusters
        of related features.


        Args:
            groups: Dict of related features groups. None to not use
                relationships. (default: None)
            use_forgotten: Create a relationship group for the forgotten
                features in the the arg groups. (default: None)
            trans_primitives: Featuretools trans primitives to use.
                None to use default. (default: None)
            max_depth: Number of iterations in the feature creation
                process. (default: 1)
            entity_set_folder_name: Folder name to store entity set with
                created features. (default: None)
            features_file_name: File name to store created features
                names. Must be JSON. (default: None)
            n_jobs: Number of parallel workers. (default: 1)
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame with new features.
        """
        # Manage groups.
        if not groups:
            groups = self._set_group(self._features)
        groups = self._fix_groups(
            features=self._features, groups=groups, use_forgotten=use_forgotten
        )

        es = self._set_entity_set(data=self._x, groups=groups)

        old_n_features = self._x.shape[1]  # For comparing later.

        if not trans_primitives:
            trans_primitives = self._TRANS_PRIMITIVES

        index_name = self._index_name(self._x)

        # Define kwargs outside the function just to improve readability.
        dfs_kwargs = {
            "entityset": es,
            "ignore_variables": {group: [index_name] for group in groups},
            "trans_primitives": trans_primitives,
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "verbose": False,
        }

        # Create features for each group.
        dfs = [ft.dfs(target_entity=key, **dfs_kwargs) for key in groups.keys()]
        # DFS returns a tuple (df and features). Split them.
        features = [features for _, features in dfs for features in features]
        dfs = [matrix for matrix, _ in dfs]

        # Concat all params from all groups to form the new dataset.
        self._x = pd.concat(dfs, axis=1)
        # Do a little cleaning just to remove useless features.
        self._x = selection.remove_low_information_features(self._x)

        # Keep only feature names that are still in the dataset.
        # noinspection PyProtectedMember
        features = [
            feature for feature in features if feature._name in self._x.columns
        ]
        # Update property.
        # noinspection PyProtectedMember
        self._features = [feature._name for feature in features]

        # Export params.
        if entity_set_folder_name:
            es.to_csv(entity_set_folder_name)
        if features_file_name:
            ft.save_features(features, features_file_name)

        # Compare number of features.
        n_new_features = self._x.shape[1] - old_n_features
        if verbose:
            print(f"{n_new_features} features created.")

        return self._x


# TODO Study feature extraction to get ideas for new methods
class Selector(_Evaluator):
    """ Automated feature selection. """

    def drop_na(
        self,
        threshold: float = 0,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Drop features that have NA ratio greater than the threshold.

        Args:
            threshold: Max ratio of NA.
            ignore: Features to ignore during dropping.
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame.
        """
        before = len(self._features)

        # Considers Inf or -Inf as NA
        pd.set_option("mode.use_inf_as_na", True)

        # Why not use the pd.DataFrame.dropna method?
        # The subset of the dropna method only works for the other axis.
        # Eg. If dropping columns, the subset is rows.
        # In this case we want to drop columns and use a subset of
        # columns. So this method has no use.

        # Get NA ratios.
        na_ratio = self._x.isna().sum() / self._x.shape[0]
        # Force the columns we want to ignore to zero.
        if ignore:
            na_ratio.loc[ignore] = 0
        # Filter and get columns names. Since it is a series, the
        # columns are indexes now.
        keep = na_ratio[na_ratio <= threshold].index

        self._x = self._x[keep]
        self._features = list(self._x.columns)

        after = len(self._features)
        if verbose:
            print(
                f"Dropped {before - after} features with NA ratio greater than "
                f"{threshold}"
            )

        return self._x

    # TODO Transform drop single in drop low variance.
    def drop_low_std(
        self,
        threshold: float = 0.05,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ):
        """
        Drop features that contains l

        Considers NA as a value.

        Args:
            threshold: Max relative standard deviation threshold.
            ignore: Features to ignore during dropping.
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame
        """
        before = len(self._features)

        std = self._x.std()
        mean = self._x.mean()
        norm_std = 2 * (std - mean).abs() / (std.abs() + mean.abs())
        norm_std = norm_std.fillna(0)
        # Force the columns we want to ignore to zero.
        if ignore:
            norm_std.loc[ignore] = np.inf
        keep = norm_std[norm_std > threshold].index

        self._x = self._x[keep]
        self._features = list(self._x.columns)

        after = len(self._features)
        if verbose:
            print(
                f"Dropped {before - after} features with normalised"
                f" standard deviation less than {threshold}"
            )

        return self._x

    def drop_correlated(
        self,
        threshold: float = 0.95,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Drop features correlated above a threshold.

        Drops from the most right to the most left column.

        Args:
            threshold: Max correlation. From 0 to 1.
            ignore: Features to ignore during dropping.
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame
        """
        before = len(self._features)

        corr = self._x.corr(method="spearman").abs()

        # Keep only the upper triangle. (k=1 exclude the diagonal).
        # Whenever applying a numpy function, turn itself into a numpy
        # array. When it happen it loses the columns names. To avoid
        # losing this information, we transform it back to DataFrame.
        upper = pd.DataFrame(np.triu(corr, k=1), columns=corr.columns)

        # Transform into a series.
        max_corr = upper.max()

        # Force the columns we want to ignore to zero.
        if ignore:
            max_corr.loc[ignore] = 0

        # Get index where max correlation attend the threshold.
        keep = max_corr[max_corr <= threshold].index

        self._x = self._x[keep]
        self._features = list(self._x.columns)

        after = len(self._features)
        if verbose:
            print(
                f"Dropped {before - after} features with correlation greater "
                f"than {threshold}"
            )

        return self._x

    def drop_single_dependence(
        self,
        threshold: float = 0.95,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Drop features with dependence from a single other feature above
        a threshold.

        Args:
            threshold: Dependence.
            ignore: Features to ignore during dropping.
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame
        """
        before = len(self._features)

        matrix = self.eval_dependence()
        matrix = matrix.drop("Dependence", axis=1)
        np.fill_diagonal(matrix.values, val=0)  # Inplace

        for idx in reversed(matrix.index):

            if ignore and idx in ignore:
                continue

            if matrix.loc[idx].max() > threshold:
                matrix = matrix.drop(idx, axis=0)
                matrix = matrix.drop(idx, axis=1)

        self._x = self._x[matrix.index]
        self._features = list(self._x.columns)

        after = len(self._features)
        if verbose:
            print(
                f"Dropped {before - after} features with dependence from a "
                f"single other feature greater than {threshold}"
            )

        return self._x

    # TODO This method run the same method several times. Instead of
    #  this, it a better a idea to replicate the rfpimp method
    #  calculation while running once.
    def drop_multiple_dependence(
        self,
        threshold: float = 0.95,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Drop features with dependence from a multiple other features
        above a threshold.

        This method uses a training iteration and will take very long
        for dataset with several features.

        Args:
            threshold: Dependence.
            ignore: Features to ignore during dropping.
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame
        """
        before = len(self._features)

        feats = self._features
        if ignore:
            feats = [feat for feat in feats if feat not in ignore]

        # Training once and removing all features would remove the
        # described and describing features at the same time.
        # Unfortunately there is no better way of doing it other than
        # removing a single feature and getting dependencies again to
        # check how did the other features change.
        # This of course will be time-consuming.

        selected = feats[:]  # Copy

        depend = self.eval_dependence(ignore)["Dependence"]

        for feat in reversed(feats):

            if depend[feat] > threshold:
                selected.remove(feat)

                # Recalculate.
                # If it is the last item from the iteration, than there
                # is no need to recalculate.

                if not feat == feats[0]:
                    depend = self.eval_dependence(ignore)["Dependence"]

        self._x = self._x[selected]
        self._features = selected

        after = len(self._features)
        if verbose:
            print(
                f"Dropped {before - after} features with dependence from "
                f"multiple other features greater than {threshold}"
            )

        return self._x

    # TODO Separate and create drop cumulative importance
    def drop_low_importance(
        self,
        threshold: float = 0.95,
        groups: Optional[Dict[str, Sequence[str]]] = None,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        n_times: int = 10,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Drop features above a cumulative importance threshold.

        Args:
            threshold: Cumulative Importance.
            groups: Groups of related features. One feature can appear
                on several groups at the same time.
            ignore: Features to ignore during dropping.
            n_times: Number of times to calculate importances. Uses the
                mean of results.
            n_jobs: Number of CPUs to use. -1 to use all available.
            verbose: Verbosity. (default: False)

        Returns:
            DataFrame
        """
        before = len(self._features)

        # Add a random column. Any feature less important than this will
        # be considered useless.
        self._x, rnd_feat_names = self._add_random_feature(self._x, 5)

        # Get importances
        imp = self.eval_importance(groups, n_times, ignore, n_jobs)

        # Remove useless features.
        rnd_feat_imp = [imp.loc[name]["Importance"] for name in rnd_feat_names]
        rnd_feat_imp = np.max(rnd_feat_imp)
        imp = imp[imp["Importance"] > rnd_feat_imp]

        # Remove based on cumulative importance threshold.
        # Keeps the feature where the threshold occurs, and remove from
        # the next on.
        idx = np.searchsorted(imp["Cumulative Importance"], threshold)
        remove = imp["Cumulative Importance"][idx + 1 :].index

        if ignore:
            remove = [rm for rm in remove if rm not in ignore]

        keep = [i for i in imp.index if i not in remove]
        self._x = self._x[keep]
        self._features = list(self._x.columns)

        after = len(self._features)
        if verbose:
            print(
                f"Dropped {before - after} features with cumulative importance "
                f"greater than {threshold}"
            )

        return self._x

    def transform(
        self,
        max_na_ratio: Optional[float] = 0,
        min_std: Optional[float] = 0.05,
        max_correlation: Optional[float] = 0.95,
        max_single_dependence: Optional[float] = 0.95,
        max_multiple_dependence: Optional[float] = 0.95,
        max_cumulative_importance: Optional[float] = 0.95,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """ Automatically select features. """

        before = len(self._features)

        if max_na_ratio is not None:
            self.drop_na(threshold=max_na_ratio, ignore=ignore, verbose=verbose)

        if min_std:
            self.drop_low_std(threshold=min_std, ignore=ignore, verbose=verbose)

        if max_correlation:
            self.drop_correlated(
                threshold=max_correlation, ignore=ignore, verbose=verbose
            )

        if max_single_dependence:
            self.drop_single_dependence(
                threshold=max_single_dependence, ignore=ignore, verbose=verbose
            )

        if max_multiple_dependence:
            self.drop_multiple_dependence(
                threshold=max_multiple_dependence,
                ignore=ignore,
                verbose=verbose,
            )

        if max_cumulative_importance:
            self.drop_low_importance(
                threshold=max_cumulative_importance,
                ignore=ignore,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        after = len(self._features)
        if verbose:
            print(
                f"Dropped a total of {before - after} features",
                f"{after} features remaining",
                sep="\n",
            )

        return self._x


# TODO Get rid of it
class Engineer(_Evaluator):
    """ Automated feature engineering. """

    def transform(
        self,
        groups: Optional[Dict[str, Sequence[str]]] = None,
        use_forgotten: bool = False,
        trans_primitives: Optional[Sequence[str]] = None,
        ignore: Optional[Union[str, Sequence[str]]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        """
        Create and select new features.

        This method do not allow much customization. For better
        customization please use the classes Creator and Selector
        separately.
        """
        # Clean initial data to avoid an excessive number of features
        # later. The main goal is to eliminate collinearity.
        # That is why low importance features are kept. We want to
        # find out if these low importance features are able to create
        # important ones.

        selector = Selector(
            data=self.data,
            target=self.target,
            features=self.features,
            task=self.task,
        )

        selector.transform(
            max_multiple_dependence=None,
            max_cumulative_importance=None,
            ignore=ignore,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        data = selector.data
        features = selector.features

        # Create features

        creator = Creator(
            data=data, target=self.target, features=features, task=self.task
        )
        creator.transform(
            groups=groups,
            use_forgotten=use_forgotten,
            trans_primitives=trans_primitives,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        data = creator.data
        features = creator.features

        # Do one final deep selection.

        selector.data = data
        selector.features = features

        selector.transform(
            max_single_dependence=None,
            max_multiple_dependence=None,
            ignore=ignore,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.data = selector.data
        self.features = selector.features
