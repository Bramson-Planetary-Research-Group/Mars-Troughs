"""
The trough model.
"""
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS

from mars_troughs import DATAPATHS


class Trough:
    def __init__(
        self,
        acc_params,
        lag_params,
        acc_model_number: int,
        lag_model_number: int,
        errorbar: float = 1.0,
        angle: float = 2.9,
        insolation_path: Union[str, Path] = DATAPATHS.INSOLATION,
        retreat_path: Union[str, Path] = DATAPATHS.RETREAT,
    ):
        """Constructor for the trough object.

        Args:
          acc_params (array like): model parameters for accumulation
          acc_model_number (int): index of the accumulation model
          lag_params (array like): model parameters for lag(t)
          lag_model_number (int): index of the lag(t) model
          errorbar (float, optional): errorbar of the datapoints in pixels; default=1
          angle (float, optional): south-facing slope angle in degrees. Default is 2.9.
          insolation_path (Union[str, Path], optional): path to the file with
            insolation data.
          retreat_path (Union[str, Path], optional): path to the file with
            retreat data
        """
        # Load in all data
        insolation, ins_times = np.loadtxt(insolation_path, skiprows=1).T
        retreats = np.loadtxt(retreat_path).T

        # Trough angle
        self.angle = angle

        # Set up the trough model
        self.acc_params = np.array(acc_params)
        self.lag_params = np.array(lag_params)
        self.acc_model_number = acc_model_number
        self.lag_model_number = lag_model_number
        self.errorbar = errorbar
        self.meters_per_pixel = np.array([500.0, 20.0])  # meters per pixel

        # Positive times are now in the past
        ins_times = -ins_times

        # Attach data to this object
        self.insolation = insolation
        self.ins_times = ins_times
        self.retreats = retreats
        
        # Set range of lag values
        self.lags = np.arange(16) + 1
        self.lags[0] -= 1
        self.lags[-1] = 20

        # Create data splines (no dependency on model parameters)
        # Insolation
        self.ins_spline = IUS(ins_times, insolation)
        self.iins_spline = self.ins_spline.antiderivative()
        self.ins2_spline = IUS(ins_times, insolation ** 2)
        self.iins2_spline = self.ins2_spline.antiderivative()
        # Retreat of ice
        self.ret_spline = RBS(self.lags, self.ins_times, self.retreats)
        self.re2_spline = RBS(self.lags, self.ins_times, self.retreats ** 2)

        # Calculate the model of lag per time 
        self.lag_model_t = self.get_lag_model_t(self.ins_times)
        
        # Calculate the model of retreat of ice per time 
        self.retreat_model_t=self.get_retreat_model_t(self.lag_model_t,self.ins_times)
        
        # Compute splines of models of lag and retreat of ice per time 
        self.compute_splines()

    def set_model(self, acc_params, lag_params, errorbar):
        """
        Updates trough model with new accumulation and lag parameters.
        Model number is kept the same for both acumulation and lag.

        Args:
            acc_params (list): Accumulation parameter(s) (same length
                                     as current acumulation parameter(s)).
            lag_params (list): Lag parameter(s) (same length
                                     as current lag parameter(s)).
            errorbar (float): Errorbar of the datapoints in pixels
        Output:
            None

        """
        assert len(acc_params) == len(self.acc_params), (
            "New and original accumulation parameters must have the same shape. %d vs %d"
            % (len(acc_params), len(self.acc_params))
        )
        assert len(lag_params) == len(self.lag_params), (
            "New and original lag parameters must have the same shape. %d vs %d"
            % (len(lag_params), len(self.lag_params))
        )
        # Set the new errorbar
        self.errorbar = errorbar
        # Set the new accumulation and lag parameters
        self.acc_params = acc_params
        self.lag_params = lag_params
        # Compute the lag at all times
        self.lag_model_t = self.get_lag_model_t(self.ins_times)
        return

    def compute_splines(self): # To be called after set_model
        """
        Computes splines of models of 1) lag per time and 
        2) retreat of ice per time. 

        Args:
            None
        Output:
            None
        """
        # lag model per time
        self.lag_model_t_spline = IUS(self.ins_times, self.lag_model_t)
        # retreat model of ice per time 
        self.retreat_model_t_spline = IUS(self.ins_times, self.retreat_model_t)
        self.iretreat_model_t_spline = self.retreat_model_t_spline.antiderivative()
        return

    def get_insolation(self, time):
        """
        Calculates the values of insolation (in W/m^2) per time.
        These values are obtained from splines of the
        times and insolation data in the Insolation.txt file.

        Args:
            time (np.ndarray): times at which we want to calculate the Insolation.
        Output:
            insolation values (np.ndarray) of the same size as time input
        """
        return self.ins_spline(time)


    def get_lag_model_t(self, time):  # Model dependent
        """
        Calculates the values of lag in mm per time.
        Lag can be constant at all times (lag = a) if model = 0
        or it can change linearly with time (lag = a + b*t) if model = 1.
        a and b are the elements of lag_params.

        Args:
            time (np.ndarray): times at which we want to calculate the lag.
        Output:
            lag values (np.ndarray) of the same size as time input
        """
        num = self.lag_model_number
        p = self.lag_params
        if num == 0:
            a = p[0]  # lag = constant
            return a * np.ones_like(time)
        if num == 1:
            a, b = p[0:2]  # lag(t) = a + b*t
            return a + b * time
        return  # error, since no number is returned
    
    def get_retreat_model_t(self, lag_t, time):
        """
        Calculates the values of retreat of ice per time.
        These values are obtained by evaluating self.ret_spline using
        the lag_model_t and time values.

        Args:
            lag_t (np.ndarray): lag per time 
            time (np.ndarray): times at which we want to calculate the retreat
        Output:
            retreat values (np.ndarray) of the same size as time input
        """
        return self.ret_spline.ev(lag_t, time)


    def get_accumulation(self, time):  # Model dependent
        """
        Calculates the values of accumulation (in m^3/W) per time.
        If model number = 0, Acc(t) = a*I(t) where I(t) is insolation at t
        If model number = 1, Acc(t) = a*I(t) + b*I(t)^2.
        a and b are the elements of acc_params.

        Args:
            time (np.ndarray): times at which we want to calculate the Acc.
        Output:
            accumulation values (np.ndarray) of the same size as time input
        """
        num = self.acc_model_number
        p = self.acc_params
        if num == 0:
            a = p[0]  # a*Ins(t)
            return -1 * (a * self.ins_spline(time))
        if num == 1:
            a, b = p[0:2]  # a*Ins(t) + b*Ins(t)^2
            return -1 * (a * self.ins_spline(time) + b * self.ins2_spline(time))
        return  # error, since no number is returned

    def get_yt(self, time: np.ndarray) -> np.ndarray:  # Model dependent
        """
        Calculates the vertical distance traveled by a point in the
        center of the high side of the trough. The vertical distance is
        calculated at each input time. This distance  is a function of the
        accumulation rate parameter H, as in dy/dt=H.

        Args:
            time (np.ndarray): times at which we want to calculate the path.
        Output:
            vertical distances (np.ndarray) of the same size as time input.
        """
        num = self.acc_model_number
        p = self.acc_params
        if num == 0:
            a = p[0]  # a*Ins(t)
            return -1 * (a * (self.iins_spline(time) - self.iins_spline(0)))
        if num == 1:
            a, b = p[0:2]  # a*Ins(t) + b*Ins(t)^2
            return -1 * (
                a * (self.iins_spline(time) - self.iins_spline(0))
                + b * (self.iins2_spline(time) - self.iins2_spline(0))
            )
        return  # error

    def get_xt(self, time: np.ndarray) -> np.ndarray:  # Model dependent
        """
        Calculates the horizontal distance traveled by a point in the
        center of the high side of the trough. The horizontal distance is
        calculated at each input time. This distance is a function of the
        accumulation rate parameter H and the retreat of ice R
        as in dx/dt=(R+Hcos(slope))/sin(slope).

        Args:
            time (np.ndarray): times at which we want the path.
        Output:
            horizontal distances (np.ndarray) of the same size as time input
        """
        yt = self.get_yt(time)
        return -self.cot_angle * yt + self.csc_angle * (
            self.iretreat_model_t_spline(time) - self.iretreat_model_t_spline(0)
        )

    def get_trajectory(self, times: Optional[np.ndarray] = None):
        """
        Obtains the x and y coordinates of the trough model as a function
        of time by concatenating the outputs of get_xt() and get_yt().

        Args:
            times (Optional[np.ndarray]): if ``None``, default to the
                times of the observed solar insolation
        Output:
            x and y coordinates (tuple) of size 2 x len(times)
        """
        if np.all(times) is None:
            x = self.get_xt(self.ins_times)
            y = self.get_yt(self.ins_times)
        else:
            x = self.get_xt(times)
            y = self.get_yt(times)
        return x, y

    @staticmethod
    def _L2_distance(x1, x2, y1, y2) -> Union[float, np.ndarray]:
        """
        The L2 (Eulerean) distance (squared) between two 2D vectors.

        Args:
            x1 (Union[float, np.ndarray]): x-coordinate of the first vector
            x2 (Union[float, np.ndarray]): x-coordinate of the second vector
            y1 (Union[float, np.ndarray]): y-coordinate of the first vector
            y2 (Union[float, np.ndarray]): y-coordinate of the second vector
        Output: L2 distance (int or float)
        """
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    def get_nearest_points(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        dist_func: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the coordinates of the nearest points between the model TMP
        and the data TMP.

        Args:
            x_data (np.ndarray): x-coordinates of the data
            y_data (np.ndarray): y-coordinatse of the data
            dist_func (Optional[Callable]): function to compute distances,
                defaults to the L2 distance
                :meth:`mars_troughs.trough.Trough._L2_distance`
        Output:
            x and y coordinates of the model TMP that are closer to the data TMP.
            (Tuple), size 2 x len(x_data)
        """
        dist_func = dist_func or Trough._L2_distance
        x_model, y_model = self.get_trajectory()
        x_out = np.zeros_like(x_data)
        y_out = np.zeros_like(y_data)
        for i, (xdi, ydi) in enumerate(zip(x_data, y_data)):
            dist = dist_func(x_model, xdi, y_model, ydi)
            ind = np.argmin(dist)
            x_out[i] = x_model[ind]
            y_out[i] = y_model[ind]
        return x_out, y_out

    def lnlikelihood(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Calculates the log-likelihood of the data given the model.
        Note that this is the natural log (ln).

        Args:
            x_data (np.ndarray): x-coordinates of the trough path
            y_data (np.ndarray): y-coordinates of the trough path
        Output:
            log-likelihood value (float)
        """
        x_model, y_model = self.get_nearest_points(x_data, y_data)
        # Variance in meters in both directions
        xvar, yvar = (self.errorbar * self.meters_per_pixel) ** 2
        chi2 = (x_data - x_model) ** 2 / xvar + (y_data - y_model) ** 2 / yvar
        return -0.5 * chi2.sum() - 0.5 * len(x_data) * np.log(xvar * yvar)

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
