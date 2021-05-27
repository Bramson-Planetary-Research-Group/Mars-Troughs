#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:02:49 2021

@author: kris
"""

from typing import List
import numpy as np
from mars_troughs.model import Model


class CustomAccuModel(Model):
    """
    Custom accumulation model
    """
    def __init__(
        self,
        times: np.ndarray,
        insolations: np.ndarray,
        intercept: float=1e-6,
        slope: float = 1e-6,
        ):

     def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Lag as a function of time

        Args:
            time (np.ndarray): times at which we want to calculate the lag.

        Output:
            np.ndarray of the same size as time input containing values of lag.
        """
        return intercept + linearCoeff*time + quadCoeff*time

     @property
     def parameter_names(self) -> List[str]:
        return ["intercept", "linearCoeff","quadCoeff"]
