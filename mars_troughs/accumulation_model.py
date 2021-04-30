"""
Model for the accumulation rates.
"""
from abc import abstractmethod

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS


class AccumulationModel():
    """
    Abstract class for computing the amount of ice accumulation.
    """

    @abstractmethod
    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AccumulationDepInsolation(AccumulationModel):
    """
    An accumulation rate model A that depends on solar insolation A(Inst(t)).
    Interpolated splines are created for the insolation as a function
    of time for faster integration.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
    """

    def __init__(self, times: np.ndarray, insolation: np.ndarray):
        
        self.ins_data_spline = IUS(times, insolation)
        self.iins_data_spline = self.ins_data_spline.antiderivative()
        self.ins2_data_spline = IUS(times, insolation ** 2)
        self.iins2_data_spline = self.ins2_data_spline.antiderivative()
        
    def get_xt(self, time: np.ndarray, iretreat_model_t_spline: np.ndarray, 
                                                       angle: float = 2.9):  
        """
        Calculates the horizontal distance x (in m) traveled by a point in the
        center of the high side of the trough. This distance x is a function of 
        the accumulation rate A and the retreat rate of ice R
        as in dx/dt=(R(l(t),t)+A(ins(t))cos(theta))/sin(theta). Where theta
        is the angle of the trough.

        Args:
            time (np.ndarray): times at which we want the path.
        Output:
            horizontal distances (np.ndarray) of the same size as time input.
        """
        self.angle=angle           
        yt = self.get_yt(time)
        
        return -self.cot_angle * yt + self.csc_angle * (
         iretreat_model_t_spline(time) - iretreat_model_t_spline(0))
    
    @property
    def angle(self) -> float:
        """
        Slope angle in degrees.
        """
        return self._angle * 180.0 / np.pi

    @angle.setter
    def angle(self, value: float) -> float:
        """Setter for the angle"""
        self._angle = value * np.pi / 180.0
        self._csc = 1.0 / np.sin(self._angle)
        self._cot = np.cos(self._angle) * self._csc

    @property
    def csc_angle(self) -> float:
        """
        Cosecant of the slope angle.
        """
        return self._csc

    @property
    def cot_angle(self) -> float:
        """
        Cotangent of the slope angle.
        """
        return self._cot

class LinearAccuIns(AccumulationDepInsolation):
    """
    Accumulation is linear in solar insolation.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
        intercept (float, optional): default is 0 m
        slope1 (float, optional): default is 1e-6 m per unit
            of solar insolation.
    """
    def __init__(self, times, insolation, intercept: float = 0.0, slope: float = 1e-6):
        
        super().__init__(times, insolation)
        
        self.intercept=intercept
        self.slope=slope
        
    def get_accumulation_at_t(self, time: np.ndarray):
        """
        Accumulation rate A (in m/year) as a  linear function of insolation: 
        A(ins(t)) = intercept + slope*ins(t).  Intercept in m/year and slope in 
        m^3/(year*W).

        Args:
            time (np.ndarray): times at which we want to calculate A, in years.
        Output:
            np.ndarray of the same size as time input containing values of 
            accumulation rates A in m/year
        
        """
        return self.intercept -1 * (self.slope * self.ins_data_spline(time))
    
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at times t traveled by a point
        in the center of the high side of the trough. This distance  is a 
        function of the accumulation rate A as y(t)=int(A(ins(t)) dt) or 
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of 
            the vertical distance y, in meters.
        
        """
        return self.intercept -1 * (self.slope * (self.iins_data_spline(time) 
                                                 - self.iins_data_spline(0)))
    
    
        
class SquareAccuIns(AccumulationDepInsolation):
    """
    Accumulation is linear in solar insolation.

    Args:
        times (np.ndarray): times at which the solar insolation is known
        insolations (np.ndarray): value of the solar insolations
        intercept (float, optional): default is 0 m
        slope1 (float, optional): default is 1e-6 m per unit
            of solar insolation.
    """
    def __init__(self, times, insolation, intercept: float = 0.0, 
                 slope1: float = 1e-6, slope2: float = 1e-6):
        
        super().__init__(times, insolation)
        
        self.intercept=intercept
        self.slope1=slope1
        self.slope2=slope2
        
    def get_accumulation_at_t(self, time: np.ndarray):
        """
        Accumulation rate A (in m/year) as a  second degree function of 
        insolation: A(ins(t)) = intercept + slope1*ins(t)+ slope2*ins(t).  
        Intercept in m/year, slope1 in m^3/(year*W) and slope2 in 
        m^5/(year*W^2).

        Args:
            time (np.ndarray): times at which we want to calculate A, in years.
        Output:
            np.ndarray of the same size as time input containing values of 
            accumulation rates A in m/year
        
        """
        return self.intercept -1 * (self.slope1 * self.ins_data_spline(time) + 
                                    self.slope2 * self.ins2_data_spline(time))
    
    def get_yt(self, time: np.ndarray):
        """
        Calculates the vertical distance y (in m) at times t traveled by a point
        in the center of the high side of the trough. This distance  is a 
        function of the accumulation rate A as y(t)=int(A(ins(t)) dt) or 
        dy/dt=A(ins(t))

        Args:
            time (np.ndarray): times at which we want to calculate y, in years.
        Output:
            np.ndarray of the same size as time input containing values of 
            the vertical distance y, in meters.
        
        """
        return self.intercept  -1 * (
               self.slope1 * (self.iins_data_spline(time) - 
                              self.iins_data_spline(0))
             + self.slope2 * (self.iins2_data_spline(time) - 
                              self.iins2_data_spline(0)))
    

# class SquareAccuIns(AccumulationDepInsolation):
#     """
#     Accumulation is two degree polynomial of insolaton.

#     Args:
#         times (np.ndarray): times at which the solar insolation is known
#         insolations (np.ndarray): value of the solar insolations
#         intercept (float, optional): default is 0 m
#         slope1 (float, optional): default is 1e-6 m per unit
#             of solar insolation.
#         slope2 (float, optional): default is 1e-6 m per unit
#             of solar insolation square.
#     """
#     def __init__(self, intercept: float = 0.0, slope1: float = 1e-6,
#                  slope2: float = 1e-6):
        
#         self.intercept=intercept
#         self.slope1=slope1
#         self.slope2=slope2
        
#     def get_accumulation_at_t(self, time: np.ndarray):
#         """
#         Accumulation as a  linear function of insolation: 
#         accu(t) = a + b*ins(t).

#         Args:
#             time (np.ndarray): times at which we want to calculate the lag.
#         Output:
#             np.ndarray of the same size as time input containing values of lag. 
        
#         """
#         return self.intercept -1 * (self.slope1 * self.ins_data_spline(time) +
#                self.slope2 * self.ins2_data_spline(time))  


