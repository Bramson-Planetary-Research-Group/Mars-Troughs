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

    @property
    def parameter_names(self) -> List[str]:
        """Returns ``["constant"]``"""
        return [self.prefix_params+"constant"]

    def eval(self, x) -> float:
        """Returns the value of :attr:`constant`."""
        constant = [val for key, val in self.parameters.items() if 'constant' in key]
        return constant * np.ones(np.size(x))


class LinearModel(Model):
    """
    A model where the output is linearly proportional to the x value.

    Args:
      intercept (float, optional): default value is 1.
      slope (float, optional): default value is 1.

      y = slope*x + intercept

    """

    @property
    def parameter_names(self) -> List[str]:
        return [self.prefix_params+"intercept", self.prefix_params+"slope"]

    def eval(self, x) -> float:
        intercept = [val for key, val in self.parameters.items() if 'intercept' in key]
        slope = [val for key, val in self.parameters.items() if 'slope' in key]
        return intercept + x * slope


class QuadModel(Model):
    """
    A model for a quadratic function of the input

    Args:
      intercept (float, optional) default is 1.0
      linearcoeff (float, optional) default is 1e-6
      quadcoeff (float, optional) default is 1e-6

      y = intercept + linearcoeff*x + quadcoeff*x^2
    """

    @property
    def parameter_names(self) -> List[str]:
        return [self.prefix_params+"intercept", self.prefix_params+ 
                "linearCoeff", self.prefix_params+"quadCoeff"]

    def eval(self, x) -> float:
        intercept = [val for key, val in self.parameters.items() if 'intercept' in key]
        linearCoeff = [val for key, val in self.parameters.items() if 'linearCoeff' in key]
        quadCoeff = [val for key, val in self.parameters.items() if 'quadCoeff' in key]
        
        p = [intercept, linearCoeff, quadCoeff]
        return sum((a * x ** i for i, a in enumerate(p)))
