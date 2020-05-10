import os
import shutil

import pandas as pd
from sklearn.datasets import load_boston

from autolearn import feature

boston = load_boston()
features = boston.feature_names
target = "MEDV"
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data[target] = boston.target


class TestLoad:
    @classmethod
    def teardown_class(cls):
        shutil.rmtree("entityset", ignore_errors=True)
        try:
            os.remove("features.json")
        except FileNotFoundError:
            pass

    def test_export_files(self):
        """ Test if files are exported accordingly. """
        feat = feature.Creator(data, target, features)
        feat.create(
            trans_primitives=["greater_than"],
            entity_set_folder_name="entityset",
            features_file_name="features.json",
        )
        feat_matrix = feature.load("entityset", "features.json")
        pd.testing.assert_frame_equal(feat.x, feat_matrix)
