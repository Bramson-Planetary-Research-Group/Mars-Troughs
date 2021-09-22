#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:14:58 2021

@author: kris
"""
from abc import abstractmethod
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from mars_troughs.generic_model import LinearModel, QuadModel, CubicModel, PowerLawModel
from mars_troughs.model import Model

class CustomAccumulationModel(Model):
    """
    Abstract class for computing the amount of ice accumulation.
    """

    prefix: str = "acc_"
    """All parameters of accumulations models start with 'acc'."""

    @abstractmethod
    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover
    
class TimeDependentAccumulationModel(CustomAccumulationModel):
    """
    An accumulation rate model that depends on the time dependent parameter
    (likely solar insolation or obliquity), A(Var(t)).
    A is in m/year. Interpolated splines are created for the parameter as
    a function of time for faster integration.

    Args:
        times (np.ndarray): times at which the variable (solar insolation, obliquity) is known
                            (in years)
        parameter (np.ndarray): values of the time dependent variable
        (solar insolation (in W/m^2), obliquity (in degrees) )
    """

    def __init__(self, times: np.ndarray, variable: np.ndarray):
        self._times = times
        self._variable = variable
        self._var_data_spline = IUS(self._times, self._variable)
        self._int_var_data_spline = self._var_data_spline.antiderivative()
        self._var2_data_spline = IUS(self._times, self._variable ** 2)
        self._int_var2_data_spline = self._var2_data_spline.antiderivative()
        self._var3_data_spline = IUS(self._times, self._variable ** 3)
        self._int_var3_data_spline = self._var3_data_spline.antiderivative()

    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Calculates the accumulation rate at times "time".

        Args:
            time (np.ndarray): times at which we want to calculate A, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            accumulation rates A, in m/year

        """
        return self.eval(self._var_data_spline(time))

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

class Linear_Obliquity(TimeDependentAccumulationModel, LinearModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
    ):
        LinearModel.__init__(self, constant, slope)
        super().__init__(obl_times, obliquity)

    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(obl(t)), dt) or
        dy/dt=A(obl(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """

        return -(
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
            )
        )

    
class Quadratic_Obliquity(TimeDependentAccumulationModel, QuadModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20,
        ):
        
        QuadModel.__init__(self, constant, slope, quad)
        super().__init__(obl_times, obliquity)
        
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """
        return -(
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
                
                + self.quad
                * (
                    self._int_var2_data_spline(time)
                    - self._int_var2_data_spline(0)
                )
            )
        )
    
class Cubic_Obliquity(TimeDependentAccumulationModel, CubicModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20,
        cubic: float =1e-30,
        ):
        
        CubicModel.__init__(self, constant, slope, quad, cubic)
        super().__init__(obl_times, obliquity)
        
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """
        return -(self.constant * time
                + 
                  (
                    self.slope
                    * (self._int_var_data_spline(time) 
                       - self._int_var_data_spline(0))
                    
                    + self.quad
                    * (self._int_var2_data_spline(time)
                        - self._int_var2_data_spline(0))
                    
                    + self.cubic
                    * (self._int_var3_data_spline(time)
                        - self._int_var3_data_spline(0))
                  )
               )

class PowerLaw_Obliquity(TimeDependentAccumulationModel, PowerLawModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        coeff: float = 1e-6,
        exponent: float = 1e-6,
        ):
        
        PowerLawModel.__init__(self, coeff, exponent)
        super().__init__(obl_times, obliquity)
        

    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """
        self._variable_exp = self._variable**self.exponent
        self._var_exp_data_spline = IUS(self._times, self._variable_exp )
        self._int_var_exp_data_spline = self._var_exp_data_spline.antiderivative()
        
        return -(self.coeff*
                     (self._int_var_exp_data_spline(time)
                     -self._int_var_exp_data_spline(0)
                     )
                )
class Linear_Insolation(TimeDependentAccumulationModel, LinearModel):
    def __init__(
        self,
        ins_times: np.ndarray,
        insolations: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
    ):
        LinearModel.__init__(self, constant, slope)
        super().__init__(ins_times, 1/insolations)

    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(obl(t)), dt) or
        dy/dt=A(obl(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """

        return -(
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
            )
        )

    
class Quadratic_Insolation(TimeDependentAccumulationModel, QuadModel):
    def __init__(
        self,
        ins_times: np.ndarray,
        insolations: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20,
        ):
        
        QuadModel.__init__(self, constant, slope, quad)
        super().__init__(ins_times, 1/insolations)
        
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """
        return -(
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
                
                + self.quad
                * (
                    self._int_var2_data_spline(time)
                    - self._int_var2_data_spline(0)
                )
            )
        )    
    
    
class Cubic_Insolation(TimeDependentAccumulationModel, CubicModel):
    def __init__(
        self,
        ins_times: np.ndarray,
        insolations: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20,
        cubic: float =1e-30,
        ):
        
        CubicModel.__init__(self, constant, slope, quad, cubic)
        super().__init__(ins_times, 1/insolations)
        
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """
        return -(self.constant * time
                + 
                  (
                    self.slope
                    * (self._int_var_data_spline(time) 
                       - self._int_var_data_spline(0))
                    
                    + self.quad
                    * (self._int_var2_data_spline(time)
                        - self._int_var2_data_spline(0))
                    
                    + self.cubic
                    * (self._int_var3_data_spline(time)
                        - self._int_var3_data_spline(0))
                  )
               )
 
class PowerLaw_Insolation(TimeDependentAccumulationModel, PowerLawModel):
    def __init__(
        self,
        ins_times: np.ndarray,
        insolations: np.ndarray,
        coeff: float = 1,
        exponent: float = -1,
        ):
        
        PowerLawModel.__init__(self, coeff, exponent)
        super().__init__(ins_times, insolations)
        

    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the accumulation rate A as y(t)=integral(A(ins(t)), dt) or
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the vertical distance y, in meters.

        """
        self._variable_exp = self._variable**self.exponent
        self._var_exp_data_spline = IUS(self._times, self._variable_exp )
        self._int_var_exp_data_spline = self._var_exp_data_spline.antiderivative()
        
        return -(self.coeff*
                     (self._int_var_exp_data_spline(time)
                     -self._int_var_exp_data_spline(0)
                     )
                )  
    
    
    
    
    
    
    
    
    
    