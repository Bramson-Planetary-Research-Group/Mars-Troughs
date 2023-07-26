"""
Abstract class for all models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Model(ABC):
    """
    Abstract class for a model, which has methods to keep track of its
    parameters and the parameters of sub-models.

    Parameters of the model **must be attributes** that match the name
    provided in the :meth:`parameter_names` property.

    Args:
      sub_models (Optional[List["Model"]]): models contained within this
        model that serve their own purpose
    """

    prefix: Optional[str] = None
    """A prefix to prepend to parameter naems when supplying them."""


    @property
    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        Names of the parameters for **this** object sorted alphabetically.
        This method **must** be implemented for all subclasses, and the names
        must consist of the names of the attributes that contain the
        parameters.

        Example:

        .. code-block:: python

           class MyModel(Model):
               a = 3

               @property
               def parameter_names(self):
                   return ["a"]

               ...

        Returns:
          names of parametes for **this** object
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        The parameters for **this** object, but not any of its sub-models.
        The parameters **must** be attributes of the object.

        Returns:
          name/value pairs where values can be numbers or arrays of numbers
        """
        return {name: getattr(self, name) for name in self.parameter_names}

    @parameters.setter
    def parameters(self, params: Dict[str, Any]) -> None:
        """
        Set the parameters for this object and all sub-models.

        Args:
          params (Dict[str, Any]): new parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
        return

   