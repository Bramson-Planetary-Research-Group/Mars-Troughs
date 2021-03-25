import os.path as op
from unittest import TestCase

from mars_troughs import DEFAULT_DATAPATH_DICT


class DatapathsTest(TestCase):
    def test_paths_exist(self):
        for path in DEFAULT_DATAPATH_DICT.values():
            assert op.exists(path)
