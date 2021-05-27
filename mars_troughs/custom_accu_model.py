#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:02:49 2021

@author: kris
"""

from typing import List
import numpy as np
from mars_troughs.model import Model
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from mars_troughs import DATAPATHS
from typing import Union
from pathlib import Path

class CustomAccuModel(Model):
    """
    Custom accumulation model
    """
    def __init__(
        self,
        coeff: float = 1e-6,
        insolation_path: Union[str, Path] = DATAPATHS.INSOLATION,
        ):
        
        insolations, times = np.loadtxt(insolation_path, skiprows=1).T
        times=-times
        
        self._ins_times=times
        self._insolations=insolations
        self.coeff=coeff
        self._inv_ins=1/self._insolations
        self._inv_ins_data_spline = IUS(self._ins_times, self._inv_ins)
        self._int_inv_ins_data_spline = self._inv_ins_data_spline.antiderivative()


    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Accumulation as a function of time

        Args:
            time (np.ndarray): times at which we want to calculate the lag.

        Output:
            np.ndarray of the same size as time input containing values of lag.
        """
        return self.coeff * self._inv_ins
    
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """

        return -self.coeff * ( self._int_inv_ins_data_spline(time) - 
                               self._int_inv_ins_data_spline(0) )
    
    def get_xt(
        self,
        time: np.ndarray,
        int_retreat_model_t_spline: np.ndarray,
        cot_angle,
        csc_angle,
    ):
        """
        Calculates the horizontal distance x (in m) traveled by a point in the
        center of the high side of the trough. This distance x is a function of
        the accumulation rate A(ins(t)) and the retreat rate of ice R(l(t),t)
        as in dx/dt=(R(l(t),t)+A(ins(t))cos(theta))/sin(theta). Where theta
        is the slope angle of the trough.

        Args:
            time (np.ndarray): times at which we want the path.
        Output:
            horizontal distances (np.ndarray) of the same size as time input, in
            meters.
        """
        yt = self.get_yt(time)

        return -cot_angle * yt + csc_angle * (
            int_retreat_model_t_spline(time) - int_retreat_model_t_spline(0)
        )
                              
    @property
    def parameter_names(self) -> List[str]:
        return ["coeff"]
