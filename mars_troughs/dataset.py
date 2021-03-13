"""
A class to read in and hold trough migration path data.
"""
from typing import Iterable, Tuple, Union

import numpy as np


class Dataset:
    """
    A class for holding the TMP data.

    Args:
        filepath (str): path to a data file with two columns, the horizontal
            and vertical components of the path
    """

    def __init__(self, filepath: str):
        x, y = np.loadtxt(filepath, unpack=True)
        self.x = x
        self.y = y
        self.N = len(x)

    def __getitem__(
        self, index: Union[Iterable, slice]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a tuple of horizontal and vertical positions of the
        TMP.

        Args:
            index (Union[Iterable, slice]): indices to select points from
                the horizontal and vertical components of the path saved
                in-memory
        """
        x = self.x[index]
        y = self.y[index]
        return (x, y)
