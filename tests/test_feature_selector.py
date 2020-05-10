""" Tests for feature.Selector. """

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

from autolearn import feature

boston = load_boston()
features = boston.feature_names
target = "MEDV"
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data[target] = boston.target


class TestSelectorInit:
    """ Test __init__ from Selector class"""

    @classmethod
    def setup_class(cls):
        cls.feat = feature.Selector(data, target, features)

    def test_init_x(self):
        """ Test if features are correctly split. """
        pd.testing.assert_frame_equal(self.feat._x, data[features])

    def test_init_y(self):
        """ Test if features are correctly split. """
        pd.testing.assert_series_equal(self.feat._y, data[target])


class TestDropNA:
    """ The the method drop_na from Selector class."""

    @classmethod
    def setup_class(cls):
        # Create a Dataframe with a column of half NA.
        na_count = data.shape[0] // 2 + 1
        cls.na_ratio = na_count / data.shape[0]
        zero_count = data.shape[0] - na_count
        cls.data = data.copy()
        cls.data["fake"] = [np.inf] * na_count + [0] * zero_count

    def test_drop(self):
        """ Test if drop class with NA ratio above the threshold."""
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_na(threshold=self.na_ratio - 0.01, ignore=None)
        after = feat._x.shape[1]
        assert before > after

    def test_ignore(self):
        """ Test if ignores a feature. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_na(threshold=self.na_ratio - 0.01, ignore="fake")
        after = feat._x.shape[1]
        assert before == after

    def test_equals_threshold(self):
        """ Test if keep feature with value equals the threshold. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_na(threshold=self.na_ratio, ignore="fake")
        after = feat._x.shape[1]
        assert before == after


class TestDropSingle:
    """ Test the method drop_single from Selector class. """

    @classmethod
    def setup_class(cls):
        # Create a Dataframe with a column with a single unique value.
        cls.data = data.copy()
        cls.data["single"] = [0] * data.shape[0]

    def test_drop_single(self):
        """ Test dropping features with single values. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_single()
        after = feat._x.shape[1]
        assert before > after

    def test_ignore(self):
        """ Test if ignores a feature. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_single(ignore="single")
        after = feat._x.shape[1]
        assert before == after


class TestDropCorrelated:
    """ Test drop_correlated method from Selector class. """

    @classmethod
    def setup_class(cls):
        # Copy first column.
        cls.data = data.copy()
        cls.data["copy"] = cls.data.iloc[:, 0]

    def test_drop(self):
        """ Test dropping features. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_correlated(threshold=0.95)
        after = feat._x.shape[1]
        assert before > after

    def test_ignore(self):
        """ Test if ignores a feature. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        # Ignore the the feature that would be dropped.
        feat.drop_correlated(threshold=0.95, ignore="copy")
        after = feat._x.shape[1]
        assert before == after


class TestDropLowImportance:
    """ Test drop_low_importance method from Selector class. """

    @classmethod
    def setup_class(cls):
        # Create a Dataframe with a column with a single unique value.
        cls.data = data.copy()
        cls.data["single"] = [0] * data.shape[0]

    def test_manage_groups(self):
        """ Test if _manage_groups created nested list. """
        groups = {"A": ["1", "2"], "B": ["3"], "C": ["4", "5", "6"]}
        features_ = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        answer = [["1", "2"], ["3"], ["4", "5", "6"], ["7"], ["8"], ["9"]]
        nested = feature.Selector._manage_groups(groups, features_)
        assert nested == answer

    def test_drop(self):
        """ Test dropping features. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_low_importance(task="regression", threshold=0.95, n_times=1)
        after = feat._x.shape[1]
        assert before > after


class TestAddRandomFeature:
    """ Test _add_random_feature method from the class Selector. """

    @classmethod
    def setup_class(cls):
        df = pd.DataFrame({"random": 0}, index=[0])
        cls.df, cls.name = feature.Selector._add_random_feature(df)

    def test_name(self):
        """ Test if it append underscore when the name exists. """
        assert self.name == "random_"

    def test_dataframe_len(self):
        """ Test if it append underscore when the name exists. """
        assert self.df.shape[1] == 2

    def test_in_dataframe(self):
        """ Test if the dataframe contains the new columns name, """
        assert "random_" in self.df.columns


class TestDropSingleDependence:
    """ Test drop_single_dependence method. """

    @classmethod
    def setup_class(cls):
        # Copy a column
        cls.data = data.copy()
        cls.data["copy"] = cls.data.iloc[:, 0]

    def test_drop_single_dependent(self):
        """ Test dropping features. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_single_dependence(threshold=0.95)
        after = feat._x.shape[1]
        assert before > after


class TestDropMultipleDependence:
    """ Test drop_multiple_dependence method. """

    @classmethod
    def setup_class(cls):
        # Copy columns
        cls.data = data.copy()
        cls.data["copy1"] = cls.data.iloc[:, 0]
        cls.data["copy2"] = cls.data.iloc[:, 0]

    def test_drop_dependent(self):
        """ Test dropping columns. """
        feat = feature.Selector(self.data, target)
        before = feat._x.shape[1]
        feat.drop_multiple_dependence(threshold=0.95)
        after = feat._x.shape[1]
        assert before > after


class TestSelect:
    """ Test select from Selector class"""

    @classmethod
    def setup_class(cls):
        cls.feat = feature.Selector(data, target, features)

    def test_select(self):
        before = self.feat._x.shape[1]
        self.feat.select(task="regression")
        after = self.feat._x.shape[1]
        assert before > after
