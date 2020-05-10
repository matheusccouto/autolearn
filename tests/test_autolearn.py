import pandas as pd
from sklearn.datasets import load_boston

import autolearn
from autolearn import feature

boston = load_boston()
features = boston.feature_names
target = "MEDV"
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data[target] = boston.target


class TestSplit:
    def test_split(self):
        """ Test split params method. """
        feat = feature.Selector(data, target, features)
        datasets = autolearn.split(feat._x, feat._y, 5000, 0.25)
        test_len = 0.25 * feat._x.shape[0]
        assert abs(datasets[1].shape[0] - test_len) < 1
