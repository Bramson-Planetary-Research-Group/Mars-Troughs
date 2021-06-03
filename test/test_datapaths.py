import os.path as op
from unittest import TestCase

import numpy as np

from mars_troughs.datapaths import DATAPATHS, load_retreat_data, load_TMP_data


class DatapathsTest(TestCase):
    def test_paths_exist(self):
        assert op.exists(DATAPATHS.DATA)
        assert op.exists(DATAPATHS.INSOLATION)
        assert op.exists(DATAPATHS.RETREAT)
        assert op.exists(DATAPATHS.TMP)

    def test_load_retreat_data(self):
        times, retreats, lags = load_retreat_data()
        assert isinstance(times, np.ndarray)
        assert isinstance(retreats, np.ndarray)
        assert isinstance(lags, np.ndarray)
        assert retreats.shape == (len(lags), len(times))

    def test_load_TMP_data(self):
        x, y = load_TMP_data()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
