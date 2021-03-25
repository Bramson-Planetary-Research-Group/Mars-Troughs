"""
General class for models.
"""

from abc import ABC
from typing import Dict


class Model(ABC):
    """
    Abstract class for a model, which has a property called `parameters`
    that returns a dictionary of the parameters in this model.
    """

    @property
    def parameters(self) -> Dict[str, float]:
        raise NotImplementedError
