"""
Model for the accumulation rates.
"""
from abc import abstractmethod
from typing import Dict

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from mars_troughs.model import Model


class AccumulationModel(Model):
    """
    Abstract class for computing the amount of ice accumulation.
    """

    @abstractmethod
    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class InsolationAccumulationModel(AccumulationModel):
    """
    An accumulation rate model that depends on solar insolation, A(Ins(t)).
    A is in m/year. Interpolated splines are created for the insolation as
    a function of time for faster integration.

    Args:
        times (np.ndarray): times at which the solar insolation is known
                            (in years)
        insolation (np.ndarray): values of the solar insolation (in W/m^2)
    """

    def __init__(self, times: np.ndarray, insolations: np.ndarray):
        self._ins_times = times
        self._insolations = insolations
        self._ins_data_spline = IUS(self._ins_times, self._insolations)
        self._int_ins_data_spline = self._ins_data_spline.antiderivative()
        self._ins2_data_spline = IUS(self._ins_times, self._insolations ** 2)
        self._int_ins2_data_spline = self._ins2_data_spline.antiderivative()

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


class LinearInsolationAccumulation(InsolationAccumulationModel):
    """
    Accumulation is linear in solar insolation.
    A(ins(t)) = intercept + slope*ins(t).
    A is in m/year.

    Args:
        times (np.ndarray): times at which the solar insolation is known
                            (in years)
        insolation values (np.ndarray): values of solar insolation (in W/m^2)
        intercept (float, optional): accumulation rate at present time.
                                     Default is 1e-6 m/year
        slope (float, optional): default is 1e-6 m/year per unit
                                 of solar insolation (m^3/(year*W)).
    """

    def __init__(
        self,
        times: np.ndarray,
        insolations: np.ndarray,
        intercept: float = 1e-6,
        slope: float = 1e-6,
    ):
        super().__init__(times, insolations)
        self.intercept = intercept
        self.slope = slope

    @property
    def parameter_names(self) -> Dict[str, float]:
        return ["intercept", "slope"]

    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        """
        Calculates the accumulation rate at times "time".

        Args:
            time (np.ndarray): times at which we want to calculate A, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            accumulation rates A, in m/year

        """
        return self.intercept + self.slope * self._ins_data_spline(time)

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
        return -(
            self.intercept * time
            + (
                self.slope
                * (self._int_ins_data_spline(time) - self._int_ins_data_spline(0))
            )
        )


class QuadraticInsolationAccumulation(InsolationAccumulationModel):
    """
    Accumulation rate A (in m/year) as a  quadratic polynomial of insolation.
    A(ins(t)) = intercept + linearCoeff*ins(t)+ quadCoeff*ins(t)^2.
    A is in m/year.

    Args:
        times (np.ndarray): times at which the solar insolation is known, in
                            years.
        insolations (np.ndarray): value of the solar insolations (in W/m^2)
        intercept (float, optional): default is 1 m/year
        linearCoeff (float, optional): default is 1e-6 m/year per unit
            of solar insolation (m^3/(year*W)).
        quadCoeff (float, optional): default is 1e-6 m/year per unit
            of solar insolation squared (m^5/(year*W^2)).
    """

    def __init__(
        self,
        times,
        insolation,
        intercept: float = 1.0,
        linearCoeff: float = 1e-6,
        quadCoeff: float = 1e-6,
    ):

        super().__init__(times, insolation)

        self.intercept = intercept
        self.linearCoeff = linearCoeff
        self.quadCoeff = quadCoeff

    @property
    def parameter_names(self) -> Dict[str, float]:
        return ["intercept", "linearCoeff", "quadCoeff"]

    def get_accumulation_at_t(self, time: np.ndarray):
        """
        Calculates the accumulation rate A at times "time".

        Args:
            time (np.ndarray): times at which we want to calculate A, in years.
        Output:
            np.ndarray of the same size as time input containing values of
            accumulation rates A, in m/year

        """
        return (
            self.intercept
            + self.linearCoeff * self._ins_data_spline(time)
            + self.quadCoeff * self._ins2_data_spline(time)
        )

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
            self.intercept * time
            + (
                self.linearCoeff
                * (self._int_ins_data_spline(time) - self._int_ins_data_spline(0))
                + self.quadCoeff
                * (
                    self._int_ins2_data_spline(time)
                    - self._int_ins2_data_spline(0)
                )
            )
        )
