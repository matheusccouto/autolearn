""" Tests for the feature.Creator """

import os
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_boston

from autolearn import feature

boston = load_boston()
features = boston.feature_names
target = "MEDV"
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data[target] = boston.target


class TestCreatorInit:
    """ Test __init__ from Creator class"""

    @classmethod
    def setup_class(cls):
        cls.feat = feature.Creator(data, target, features)

    def test_init_x(self):
        """ Test if features are correctly split. """
        pd.testing.assert_frame_equal(self.feat.x, data[features])

    def test_init_y(self):
        """ Test if features are correctly split. """
        pd.testing.assert_series_equal(self.feat.y, data[target])


class TestGroupManagement:
    """ Test managing groups from Creator class"""

    @classmethod
    def setup_class(cls):
        cls.feat = feature.Creator(data, target, features)

    def test_set_group(self):
        """ Test _set_group method. """
        dict_ = self.feat._set_group(self.feat.features)
        np.testing.assert_equal(dict_["whole"], self.feat.features)

    def test_get_extra_feats(self):
        """ Test _get_extra_feats method. """
        fake_feat = ["Fake Feature"]
        group = {"whole": list(self.feat.features) + fake_feat}
        extra = self.feat._get_extra_feats(self.feat.features, group)
        np.testing.assert_equal(extra, extra)

    def test_get_missing_feats(self):
        """ Test _get_missing_feats method. """
        group = {"whole": self.feat.features[:-1]}
        missing = self.feat._get_missing_feats(self.feat.features, group)
        np.testing.assert_equal(missing, self.feat.features[-1:])

    def test_fix_groups_raises_extra(self):
        """ Test that _fix_groups raises when it finds extra features. """
        with pytest.raises(ValueError):
            fake_feat = ["Fake Feature"]
            group = {"whole": list(self.feat.features) + fake_feat}
            self.feat._fix_groups(self.feat.features, group)

    def test_fix_groups_forgotten(self):
        """ Test that _fix_groups creates the forgotten key. """
        group = {"whole": self.feat.features[:-1]}
        fixed = self.feat._fix_groups(
            self.feat.features, group, use_forgotten=True
        )
        assert fixed["forgotten"] == self.feat.features[-1:]


class TestEntitySet:
    """ Test the entity set from the Creator class. """

    @classmethod
    def setup_class(cls):
        cls.feat = feature.Creator(data, target, features)

    def test_index_name(self):
        """ Test get index name when name is not None. """
        df = pd.DataFrame()
        df.index.name = r"¯\_(ツ)_/¯"
        index_name = feature.Creator._index_name(df)
        assert index_name == r"¯\_(ツ)_/¯"

    def test_index_name_default(self):
        """ Test get index name when name is None. """
        df = pd.DataFrame()
        index_name = feature.Creator._index_name(df)
        assert index_name == "index"

    def test_set_entity_set(self):
        """ Test setting an entity set"""
        groups = {"p1": self.feat.features[:5], "p2": self.feat.features[5:]}
        es = self.feat._set_entity_set(data, groups)
        assert len(es.entities) == 2


class TestCreate:
    @classmethod
    def teardown_class(cls):
        shutil.rmtree("entityset", ignore_errors=True)
        try:
            os.remove("features.json")
        except FileNotFoundError:
            pass

    def test_create(self):
        """ Test creating variables. """
        feat = feature.Creator(data, target, features)
        old_n_cols = len(feat.x.columns)
        feat.create()
        new_n_cols = len(feat.x.columns)
        assert new_n_cols > old_n_cols

    def test_custom_trans_primitives(self):
        """ Test changing the default trans primitives. """
        feat = feature.Creator(data, target, features)
        feat.create(trans_primitives=["add_numeric"])
        custom_n_cols = len(feat.x.columns)

        feat = feature.Creator(data, target, features)
        feat.create(trans_primitives=None)
        default_n_cols = len(feat.x.columns)
        assert default_n_cols > custom_n_cols

    def test_custom_max_depth(self):
        """ Test changing the default trans primitives. """
        feat = feature.Creator(data, target, features)
        feat.create(trans_primitives=["greater_than", "and"], max_depth=1)
        depth1_n_cols = len(feat.x.columns)

        feat = feature.Creator(data, target, features)
        feat.create(trans_primitives=["greater_than", "and"], max_depth=2)
        depth2_n_cols = len(feat.x.columns)

        assert depth2_n_cols > depth1_n_cols

    def test_export_files(self):
        """ Test if files are exported accordingly. """
        feat = feature.Creator(data, target, features)
        feat.create(
            trans_primitives=["greater_than"],
            entity_set_folder_name="entityset",
            features_file_name="features.json",
        )
        assert os.path.exists("entityset") and os.path.exists("features.json")

    def test_not_single_value(self):
        """ Test that no single values features are created. """
        feat = feature.Creator(data, target, features)
        feat.create(trans_primitives=["greater_than"])
        uniques_count = [len(feat.x[col].unique()) for col in feat.x.columns]
        assert 1 not in uniques_count

    def test_not_all_nan(self):
        """ Test that no NA features are created. """
        feat = feature.Creator(data, target, features)
        feat.create(trans_primitives=["greater_than"])
        before = len(feat.x.columns)
        after = len(feat.x.dropna(1, how="all").columns)
        assert before == after
