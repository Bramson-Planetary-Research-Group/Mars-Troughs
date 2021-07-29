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
        self.sub_models: List[Model] = sub_models or []

        # Check for duplicate names between here and the sub-models
        our_names = set(self.parameter_names)
        for sub_model in self.sub_models:
            their_names = set(sub_model.all_parameter_names)
            duplicates = our_names.intersection(their_names)
            assert not duplicates, f"duplicate parameter names {duplicates}"

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

    @property
    def all_parameter_names(self) -> List[str]:
        """
        Names of the parameters for this object and all sub-models.

        Returns:
          names of parametes for this object and all sub-models
        """
        return self.all_parameters.keys()

    @property
    def all_parameters(self) -> Dict[str, Any]:
        """
        The parameters for this model and all sub-models.

        Returns:
          key-value pairs for this model and all sub-models
        """
        # Find parameters of sub_models
        sub_pars = {}
        for i, sub_model in enumerate(self.sub_models):
            # If the sub model doesn't have a prefix then prepend a number
            sub_prefix = sub_model.prefix or str(i) + "_"
            # Attach the prefix of submodels to their keys
            for key, value in sub_model.all_parameters.items():
                sub_pars[sub_prefix + key] = value

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
        for i, sub_model in enumerate(self.sub_models):
            # Pull out only the parameters associated with the sub model,
            # based on its prefix (or the default assigned prefix)
            # Set the sub model's `all_parameters` to be this parameter subset
            sub_prefix = sub_model.prefix or str(i) + "_"
            sub_params = {
                k[len(sub_prefix) :]: v
                for k, v in params.items()
                if k.startswith(sub_prefix)
            }
            sub_model.all_parameters = sub_params
        return
