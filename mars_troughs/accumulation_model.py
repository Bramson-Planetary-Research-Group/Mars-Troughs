"""
Model for the accumulation rates.
"""
from abc import abstractmethod
from typing import Dict

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from mars_troughs.generic_model import LinearModel, QuadModel
from mars_troughs.model import Model


class AccumulationModel(Model):
    """
    Abstract class for computing the amount of ice accumulation.
    """

    prefix: str = "acc_"
    """All parameters of accumulations models start with 'acc'."""

    @abstractmethod
    def get_accumulation_at_t(self, time: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


class TimeDependentAccumulationModel(AccumulationModel):
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
        dt = np.diff(times)
        assert (dt > 0).all(), "times must be monotonically increasing"
        self._times = times
        self._variable = variable
        self._var_data_spline = IUS(self._times, self._variable)
        self._int_var_data_spline = self._var_data_spline.antiderivative()
        self._var2_data_spline = IUS(self._times, self._variable ** 2)
        self._int_var2_data_spline = self._var2_data_spline.antiderivative()

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
        return self.eval_integral(self._var_data_spline(time))

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


class LinearInsolation(TimeDependentAccumulationModel, LinearModel):
    """
    Accumulation is linear in solar insolation.
    A(ins(t)) = constant + slope*ins(t).
    A is in m/year.

    Args:
        times (np.ndarray): times at which the solar insolation is known
                            (in years)
        insolation values (np.ndarray): values of solar insolation (in W/m^2)
        constant (float, optional): accumulation rate at present time.
                                     Default is 1e-6 m/year
        slope (float, optional): default is 1e-6 m/year per unit
                                 of solar insolation (m^3/(year*W)).
    """

    def __init__(
        self,
        times: np.ndarray,
        insolations: np.ndarray,
        constant: float = 1e-6,
        slope: float = 1e-6,
    ):
        super().__init__(times, insolations)
        LinearModel.__init__(self, constant=constant, slope=slope)

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
            self.constant * time
            + (
                self.slope
                * (self._int_var_data_spline(time) - self._int_var_data_spline(0))
            )
        )


class QuadraticInsolation(TimeDependentAccumulationModel, QuadModel):
    """
    Accumulation rate A (in m/year) as a  quadratic polynomial of insolation.
    A(ins(t)) = constant + slope*ins(t)+ quad*ins(t)^2.
    A is in m/year.

    Args:
        times (np.ndarray): times at which the solar insolation is known, in
                            years.
        insolations (np.ndarray): value of the solar insolations (in W/m^2)
        constant (float, optional): default is 1 m/year
        slope (float, optional): default is 1e-6 m/year per unit
            of solar insolation (m^3/(year*W)).
        quad (float, optional): default is 1e-6 m/year per unit
            of solar insolation squared (m^5/(year*W^2)).
    """

    def __init__(
        self,
        times,
        insolation,
        constant: float = 1.0,
        slope: float = 1e-6,
        quad: float = 1e-6,
    ):
        super().__init__(times, insolation)
        QuadModel.__init__(self, constant=constant, slope=slope, quad=quad)

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
        delta = self._int_var_data_spline(time) - self._int_var_data_spline(0)
        delta2 = self._int_var2_data_spline(time) - self._int_var2_data_spline(0)
        return -(
            self.constant * time + (self.slope * (delta) + self.quad * (delta2))
        )


class LinearObliquity(TimeDependentAccumulationModel, LinearModel):
    def __init__(
        self,
        obl_times: np.ndarray,
        obliquity: np.ndarray,
        constant: float = 1.0,
        slope: float = 1.0,
    ):
        LinearModel.__init__(self, constant=constant, slope=slope)
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


ACCUMULATION_MODEL_MAP: Dict[str, Model] = {
    "linear": LinearInsolation,
    "quadratic": QuadraticInsolation,
    "obliquity": LinearObliquity,
}
