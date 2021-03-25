"""
Tests of the ``Dataset`` class used to hold TMPs.
"""
import os.path as op
import tempfile
from unittest import TestCase

import numpy as np
import pytest

from mars_troughs import Dataset


class DatasetTest(TestCase):
    def setUp(self):
        # Create a phony TMP that is just a list of integers
        self.N_data = 20  # number of fake data points
        self.x = np.arange(self.N_data)
        self.y = self.x * 2.0
        # Bundle it together
        data = np.array([self.x, self.y]).T  # shape (N_data, 2)

        # Make a temporary file to put our fake TMP
        tempdir = tempfile.mkdtemp()
        self.filepath = op.join(tempdir, "fake_TMP.txt")
        np.savetxt(self.filepath, data)

    def test_smoke(self):
        # Test that the dataset can be created and it reads
        # in the data correctly
        ds = Dataset(filepath=self.filepath)
        assert ds is not None
        assert ds.N == self.N_data
        assert (ds.x == self.x).all()
        assert (ds.y == self.y).all()

    def test_assert_missing_file(self):
        with pytest.raises(Exception):
            _ = Dataset(filepath="a garbage file path")

    def test_getitem(self):
        # Test the __getitem__ method
        ds = Dataset(filepath=self.filepath)

        # A single index
        x, y = ds[0]
        assert x == 0
        assert y == 0

        # A slice
        x, y = ds[:5]
        assert len(x) == 5
        assert len(y) == 5
