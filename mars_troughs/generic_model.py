#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:37:51 2021

@author: kristel Iz.

Linear model for both accumulation and lag
"""
from numbers import Number
from typing import List, Union

import numpy as np

from mars_troughs.model import Model


class ConstantModel(Model):
    """
    A model where the parameter is a constant value.
    The :meth:`eval` method returns the :attr:`constant`
    attribute.

    Args:
      constant (float, optional): default value is 1. The
        constant value returned by :meth:`eval`. It is a
        parameter.
    """

    def __init__(self, constant: float = 1.0):
        self.constant = constant
        #super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        """Returns ``["constant"]``"""
        return ["constant"]

    def eval(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        """Returns the value of :attr:`constant`."""
        return self.constant + x * 0


class LinearModel(Model):
    """
    A model where the output is linearly proportional to the x value.

    Args:
      constant (float, optional): default value is 1.
      slope (float, optional): default value is 1.

      y = slope*x + constant

    """

    def __init__(self, constant: float = 1.0, slope: float = 1.0):
        self.constant = constant
        self.slope = slope
        #super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        return ["constant", "slope"]

    def eval(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.constant + self.slope * x


class QuadModel(Model):
    """
    A model for a quadratic function of the input

    Args:
      constant (float, optional) default is 1.0
      slope (float, optional) default is 1e-6
      quadcoeff (float, optional) default is 1e-6

      y = constant + linearcoeff*x + quadcoeff*x^2
    """

    def __init__(
        self,
        constant: float = 1.0,
        slope: float = 1e-6,
        quad: float = 1e-6,
    ):
        self.constant = constant
        self.slope = slope
        self.quad = quad
        #super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        return [
            "constant",
            "slope",
            "quad",
        ]

    def eval(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.quad * x ** 2 + self.slope * x + self.constant

class CubicModel(Model):
    """
    A model for a cubic function of the input
    
    Args:
      constant (float, optional) default is 1.0
      slope (float, optional) default is 1e-6
      quad (float, optional) default is 1e-6 
      cubic (float, optional) defatul is 1e-6
   
    y = constant + slope * x + quad * x^2 + cubic * x^3
    """
    
    def __init__(
        self,
        constant: float = 1.0,
        slope: float = 1e-6,
        quad: float = 1e-6,
        cubic: float = 1e-6,
    ):
        self.constant = constant
        self.slope = slope
        self.quad = quad
        self.cubic = cubic
        #super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        return [
            "constant",
            "slope",
            "quad",
            "cubic"
        ]
    
    def eval(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.cubic * x ** 3 + self.quad * x ** 2 + self.slope * x + self.constant
    
class PowerLawModel(Model):
    """
    A model for a power law function of the input
    
    Args:
      coeff (float, optional) default is 1.0
      exponent (float, optional) default is 1.0
      
    y = coeff * x^exponent
    """
    
    def __init__(
        self,
        coeff: float = 1.0,
        exponent: float = 1.0,
    ):
        self.coeff=coeff
        self.exponent=exponent
        #super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        return [
            "coeff",
            "exponent",
        ]
    
    def eval(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        return self.coeff * x ** self.exponent

      
      
      
      
      
      
      
    
