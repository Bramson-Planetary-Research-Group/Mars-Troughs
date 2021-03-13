"""
A class to read in and hold trough migration path data.
"""
from typing import Iterable, Tuple, Union

import numpy as np


class Dataset:
    def __init__(self, filepath: str):
        x, y = np.loadtxt(filepath, unpack=True)
        self.x = x
        self.y = y
        self.N = len(x)

    def __getitem__(
        self, index: Union[Iterable, slice]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x[index]
        y = self.y[index]
        return (x, y)
