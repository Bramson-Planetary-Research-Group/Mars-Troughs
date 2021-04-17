import os.path as op
from unittest import TestCase

from mars_troughs import DATAPATHS


class DatapathsTest(TestCase):
    def test_paths_exist(self):
        assert op.exists(DATAPATHS.DATA)
        assert op.exists(DATAPATHS.INSOLATION)
        assert op.exists(DATAPATHS.RETREAT)
        assert op.exists(DATAPATHS.TMP)
