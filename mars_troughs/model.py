"""
General class for models.
"""

from abc import ABC
from typing import Dict, List

from dataclass import asdict, dataclass, field


@dataclass
class Model(ABC):
    """
    Abstract class for a model, which has a property called `parameters`
    that returns a dictionary of the parameters in this model.
    """

    prefix: str = ""
    sub_models: List["Model"] = field(default_factory=list)

    def __post_init__(self) -> None:
        our_names = set(self.parameter_names)
        for sub_model in self.sub_models:
            their_names = set(sub_model.parameter_names)
            duplicates = our_names.intersection(their_names)
            assert not duplicates, f"duplicate parameter names {duplicates}"

    @property
    def paramter_names(self) -> List[str]:
        return sorted(self.parameters.keys())

    @property
    def parameters(self) -> Dict[str, float]:
        # Find parameters of sub_models
        sub_pars = {}
        for sub_model in self.sub_models:
            sub_pars = {**sub_pars, **sub_model.parameters}

        # Convert our attributes to a dict
        our_pars = asdict(self)

        # Remove the 'sub_models' key from our parameters list
        _ = our_pars.pop("sub_models")

        # Prepend our prefix
        if self.prefix != "":
            our_pars = {
                f"{self.prefix}_" + key: value for key, value in our_pars.items()
            }

        # Bundle up our parameters and all sub model's parameters
        return {**our_pars, **sub_pars}
