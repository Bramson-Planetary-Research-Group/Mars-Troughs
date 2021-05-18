#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:37:51 2021

@author: laferrierek

Linear model for both accumulation and lag
"""
from typing import List

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

    @property
    def parameter_names(self) -> List[str]:
        """Returns ``["constant"]``"""
        return ["constant"]

    def eval(self, *args) -> float:
        """Returns the value of :attr:`constant`."""
        return self.constant


# TODO - write an __init__ method
class LinearModel(Model):
    intercept: float = 1.0
    slope: float = 1.0

    @property
    def parameter_names(self) -> List[str]:
        return ["intercept", "slope"]

    def eval(self, x) -> float:
        return self.intercept + x * self.slope


# TODO - write an __init__ method
class QuadModel(Model):
    intercept: float = 1.0
    linearCoeff: float = 1e-6
    quadCoeff: float = 1e-6

    @property
    def parameters_names(self) -> List[str]:
        return ["intercept", "linearCoeff", "quadCoeff"]

    def eval(self, x) -> float:
        p = [self.intercept, self.linearCoeff, self.quadCoeff]
        return sum((a * x ** i for i, a in enumerate(p)))
