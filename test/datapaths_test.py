import os.path as op
from unittest import TestCase

import numpy as np

from mars_troughs.datapaths import DATAPATHS, load_retreat_data, load_TMP_data


class DatapathsTest(TestCase):
    def test_paths_exist(self):
        assert op.exists(DATAPATHS.DATA)
        assert op.exists(DATAPATHS.INSOLATION1)
        assert op.exists(DATAPATHS.INSOLATION2)
        assert op.exists(DATAPATHS.RETREAT)
        assert op.exists(DATAPATHS.TMP1)
        assert op.exists(DATAPATHS.TMP2)
        assert op.exists(DATAPATHS.OBLIQUITY)

    def test_load_retreat_data(self):
        times, retreats, lags = load_retreat_data()
        assert isinstance(times, np.ndarray)
        assert isinstance(retreats, np.ndarray)
        assert isinstance(lags, np.ndarray)
        assert retreats.shape == (len(lags), len(times))
        assert lags[0] == 1
        assert lags[-1] == 20

    def test_load_TMP_data(self):
        x, y = load_TMP_data(1)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        x2, y2 = load_TMP_data(2)
        assert isinstance(x2, np.ndarray)
        assert isinstance(y2, np.ndarray)
        assert x2.shape == y2.shape
