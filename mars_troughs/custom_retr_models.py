#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom retreat models

"""
import numpy as np
from abc import abstractmethod
from mars_troughs.model import Model
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from mars_troughs.generic_model import (ConstantModel, 
                                        LinearModel,
                                        QuadModel, 
                                        CubicModel, 
                                        PowerLawModel)

class RetreatModel(Model):
    """
    Abstract class for retreat models, that have a method
    called :meth:`get_retreat_at_t` that returns retreat lag
    as a function of time.
    """

    prefix: str = "retr_"
    """All parameters of retreat models start with 'retreat'."""
    @abstractmethod
    def get_retreat_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Retreat as a function of time

        Args:
            time (np.ndarray): times at which we want to calculate the retreat.

        Output:
            np.ndarray of the same size as time input containing values of retreat.
        """
        #return self.eval(time)
        raise NotImplementedError # this comes from Acc model
   
class TimeDependentRetreatModel(RetreatModel):
    """
    A retreat rate model that depends on the time dependent parameter
    (likely solar insolation or obliquity), R(Var(t)).
    R is in m/year. Interpolated splines are created for the parameter as
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

    def get_retreat_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Calculates the retreat rate at times "time".

        Args:
            time (np.ndarray): times at which we want to calculate R, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            retreat rates R, in m/year

        """
        
        
        return self.eval(self._var_data_spline(time))
        

class Constant_Retreat(TimeDependentRetreatModel, ConstantModel):
    """
    The retreat rate is constant and does not depend on time.

    Args:
        constant (float, optional): default is 1 millimeter. The lag
            thickness at all times.
    """

    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1e-6,
    ):
        super().__init__(obl_times, obliquity)  # note: `super` maps to the LagModel parent class
        ConstantModel.__init__(self, constant=constant)

    def get_rt(self, time: np.ndarray):
        """
        Calculates the retreat distance r (in m) traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the retreat rate A as r(t)=integral(AR(obl(t)), dt) or
        dy/dt=R(obl(t))

        Args:
            time (np.ndarray): times at which we want to calculate r, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the retreat distance r, in meters.

        """

        return (self.constant * time)
              

class Linear_RetreatO(TimeDependentRetreatModel, LinearModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1e-6,#1e-6, # was e-6, then e-5
        slope: float = 1e-8, #1e-8, # was e-8, then 1e-6
    ):
        LinearModel.__init__(self, constant, slope)
        super().__init__(obl_times, obliquity)

    def get_rt(self, time: np.ndarray):
        """
        Calculates the retreat distance r (in m) traveled by a point
        in the center of the high side of the trough. This distance  is a
        function of the retreat rate R as r(t)=integral(R(obl(t)), dt) or
        dy/dt=R(obl(t))

        Args:
            time (np.ndarray): times at which we want to calculate r, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            the retreat distance r, in meters.

        """
        
        
       
        return (
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
            )
        )
 

    

class Quadratic_RetreatO(TimeDependentRetreatModel,QuadModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
        quad: float = 1e-20, # was 1e-20
        ):
        
        QuadModel.__init__(self, constant, slope, quad)
        super().__init__(obl_times, obliquity)
        
    def get_rt(self, time: np.ndarray):
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
        output = (
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
        
        
        return output
            
       

class Cubic_RetreatO(TimeDependentRetreatModel, CubicModel):
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
        
    def get_rt(self, time: np.ndarray):
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
        return (self.constant * time
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

class PowerLaw_RetreatO(TimeDependentRetreatModel, PowerLawModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        coeff: float = 0.1,
        exponent: float = -2,
        ):
        
        PowerLawModel.__init__(self, coeff, exponent)
        super().__init__(obl_times, obliquity)
        

    def get_rt(self, time: np.ndarray):
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
        
        return (self.coeff*
                     (self._int_var_exp_data_spline(time)
                     -self._int_var_exp_data_spline(0)
                     )
                )
    
#%% insolation functions
  
class Linear_RetreatI(TimeDependentRetreatModel, LinearModel):
    def __init__(
        self,
        ins_times: np.ndarray,
        insolations: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-8,
    ):
        LinearModel.__init__(self, constant, slope)
        super().__init__(ins_times, 1/insolations)

    def get_rt(self, time: np.ndarray):
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

        return (
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
            )
        )
        
        
 
class Quadratic_RetreatI(TimeDependentRetreatModel, QuadModel):
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
        
    def get_rt(self, time: np.ndarray):
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
        return (
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
    
    
class Cubic_RetreatI(TimeDependentRetreatModel, CubicModel):
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
        
    def get_rt(self, time: np.ndarray):
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
        return (self.constant * time
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
 
class PowerLaw_RetreatI(TimeDependentRetreatModel, PowerLawModel):
    def __init__(
        self,
        ins_times: np.ndarray,
        insolations: np.ndarray,
        coeff: float = 0.1,
        exponent: float = -1,
        ):
        
        PowerLawModel.__init__(self, coeff, exponent)
        super().__init__(ins_times, insolations)
        

    def get_rt(self, time: np.ndarray):
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
        
        return (self.coeff*
                     (self._int_var_exp_data_spline(time)
                     -self._int_var_exp_data_spline(0)
                     )
                )         
    
    