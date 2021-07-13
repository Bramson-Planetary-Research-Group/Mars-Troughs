#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:37:51 2021

@author: laferrierek

Linear model for both accumulation and lag
"""
from typing import List

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
        super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        """Returns ``["constant"]``"""
        return ["constant"]

    def eval(self, x) -> float:
        """Returns the value of :attr:`constant`."""
        return self.constant * np.ones(np.size(x))


class LinearModel(Model):
    """
    A model where the output is linearly proportional to the x value.

    Args:
      intercept (float, optional): default value is 1.
      slope (float, optional): default value is 1.

      y = slope*x + intercept

    """

    def __init__(self, intercept: float = 1.0, slope: float = 1.0):
        self.intercept = intercept
        self.slope = slope
        super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        return ["intercept", "slope"]

    def eval(self, x) -> float:
        return self.intercept + x * self.slope


class QuadModel(Model):
    """
    A model for a quadratic function of the input

    Args:
      intercept (float, optional) default is 1.0
      linearcoeff (float, optional) default is 1e-6
      quadcoeff (float, optional) default is 1e-6

      y = intercept + linearcoeff*x + quadcoeff*x^2
    """

    def __init__(
        self,
        intercept: float = 1.0,
        linearCoeff: float = 1e-6,
        quadCoeff: float = 1e-6,
    ):
        self.intercept = intercept
        self.linearCoeff = linearCoeff
        self.quadCoeff = quadCoeff
        super().__init__()

    @property
    def parameter_names(self) -> List[str]:
        return [
            "intercept",
            "linearCoeff",
            "quadCoeff",
        ]

    def eval(self, x) -> float:
        p = [self.intercept, self.linearCoeff, self.quadCoeff]
        return sum((a * x ** i for i, a in enumerate(p)))
