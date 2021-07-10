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

    prefix: str = ""
    """A prefix to prepend to parameter naems when supplying them."""

    def __init__(self, sub_models: Optional[List["Model"]] = None) -> None:
        # Add sub_models as an attribute
        self.sub_models = sub_models or []

        # Check for duplicate names between here and the sub-models
        our_names = set(self.parameter_names)
        for sub_model in self.sub_models:
            their_names = set(sub_model.all_parameter_names)
            duplicates = our_names.intersection(their_names)
            assert not duplicates, f"duplicate parameter names {duplicates}"

    @property
    def prefix_length(self) -> int:
        """Length of the prefix for parameter names of this model."""
        return len(self.prefix)

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
          names of parametes for **this** object, possibly with a prefix
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        The parameters for **this** object, but not any of its sub-models.
        The parameters **must** be attributes of the object.

        .. note:: The ``prefix`` attribute will be prepended to keys.

        Returns:
          name/value pairs where values can be numbers or arrays of numbers
        """
        return {
            self.prefix + name: getattr(self, name)
            for name in self.parameter_names
        }

    @parameters.setter
    def parameters(self, params: Dict[str, Any]) -> None:
        """
        Set the parameters for this object and all sub-models.
        This setter accepts keys that do or don't start with
        the prefix.

        Args:
          params (Dict[str, Any]): new parameters
        """
        for key, value in params.items():
            key = (
                key[self.prefix_length :] if key.startswith(self.prefix) else key
            )
            setattr(self, key, value)
        return

    @property
    def all_parameter_names(self) -> List[str]:
        """
        Names of the parameters for this object and all sub-models.

        Returns:
          names of parametes for this object and all sub-models
        """
        return sorted(self.all_parameters.keys())

    @property
    def all_parameters(self) -> Dict[str, Any]:
        """
        The parameters for this model and all sub-models.

        Returns:
          key-value pairs for this model and all sub-models
        """
        # Find parameters of sub_models
        sub_pars = {}
        for sub_model in self.sub_models:
            sub_pars = {**sub_pars, **sub_model.all_parameters}

        # Bundle up our parameters and all sub model's parameters
        return {**self.parameters, **sub_pars}

    @all_parameters.setter
    def all_parameters(self, params: Dict[str, Any]) -> None:
        """
        Setter for this model's parameters and all sub-models.

        Args:
          params (Dict[str, Any]): new parameters
        """
        # Update this model's parameters
        self.parameters = {key: params[key] for key in self.parameter_names}

        # Update sub model's parameters
        for sub_model in self.sub_models:
            sub_model.all_parameters = {
                key: params[key] for key in sub_model.all_parameter_names
            }
        return
