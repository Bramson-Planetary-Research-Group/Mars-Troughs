import importlib.resources as pkg_resources

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS


class Trough(object):
    def __init__(
        self,
        acc_params,
        lag_params,
        acc_model_number,
        lag_model_number,
        errorbar=1.0,
    ):
        """Constructor for the trough object.

        Args:
            acc_params (array like): model parameters for accumulation
            acc_model_number (int): index of the accumulation model
            lag_params (array like): model parameters for lag(t)
            lag_model_number (int): index of the lag(t) model
            errorbar (float): errorbar of the datapoints in pixels; default=1
        """
        # Load in all data
        with pkg_resources.path(__package__, "Insolation.txt") as path:
            insolation, ins_times = np.loadtxt(path, skiprows=1).T
        with pkg_resources.path(__package__, "R_lookuptable.txt") as path:
            retreats = np.loadtxt(path).T
        with pkg_resources.path(__package__, "TMP_xz.txt") as path:
            xdata, ydata = np.loadtxt(path, unpack=True)
            # TODO: remember what this means... lol
            # I'm pretty sure one file has temp data and the other
            # has real data.
            # xdata, ydata = np.loadtxt(here+"/RealXandZ.txt")

        # Trough angle
        self.angle_degrees = 2.9  # degrees
        self.sin_angle = np.sin(self.angle_degrees * np.pi / 180.0)
        self.cos_angle = np.cos(self.angle_degrees * np.pi / 180.0)
        self.csc_angle = 1.0 / self.sin_angle
        self.sec_angle = 1.0 / self.cos_angle
        self.tan_angle = self.sin_angle / self.cos_angle
        self.cot_angle = 1.0 / self.tan_angle
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
        self.xdata = xdata * 1000  # meters
        self.ydata = ydata  # meters
        self.Ndata = len(self.xdata)  # number of data points

        # Create splines
        self.lags = np.arange(16) + 1
        self.lags[0] -= 1
        self.lags[-1] = 20
        self.ins_spline = IUS(ins_times, insolation)
        self.iins_spline = self.ins_spline.antiderivative()
        self.ins2_spline = IUS(ins_times, insolation ** 2)
        self.iins2_spline = self.ins2_spline.antiderivative()
        self.ret_spline = RBS(self.lags, self.ins_times, self.retreats)
        self.re2_spline = RBS(self.lags, self.ins_times, self.retreats ** 2)

        # Pre-calculate the lags at all times
        self.lags_t = self.get_lag_at_t(self.ins_times)
        self.compute_splines()

    def set_model(self, acc_params, lag_params, errorbar):
        """Setup a new model, with new accumulation and lag parameters.
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
        self.lags_t = self.get_lag_at_t(self.ins_times)
        return

    def compute_splines(self):
        # To be called after set_model
        self.lag_spline = IUS(self.ins_times, self.lags_t)
        self.Retreat_model_at_t = self.get_Rt_model(self.lags_t, self.ins_times)
        self.retreat_model_spline = IUS(self.ins_times, self.Retreat_model_at_t)
        self.iretreat_model_spline = self.retreat_model_spline.antiderivative()
        return

    def get_insolation(self, time):
        return self.ins_spline(time)

    def get_retreat(self, lag, time):
        return self.ret_spline(time, lag)

    def get_lag_at_t(self, time):  # Model dependent
        num = self.lag_model_number
        p = self.lag_params
        if num == 0:
            a = p[0]  # lag = constant
            return a * np.ones_like(time)
        if num == 1:
            a, b = p[0:2]  # lag(t) = a + b*t
            return a + b * time
        return  # error, since no number is returned

    def get_Rt_model(self, lags, times):
        ret = self.ret_spline.ev(lags, times)
        return ret

    def get_accumulation(self, time):  # Model dependent
        num = self.acc_model_number
        p = self.acc_params
        if num == 0:
            a = p[0]  # a*Ins(t)
            return -1 * (a * self.ins_spline(time))
        if num == 1:
            a, b = p[0:2]  # a*Ins(t) + b*Ins(t)^2
            return -1 * (a * self.ins_spline(time) + b * self.ins2_spline(time))
        return  # error, since no number is returned

    def get_yt(self, time):  # Model dependent
        # This is the depth the trough has traveled
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

    def get_xt(self, time):
        # This is the horizontal distance the trough has traveled
        # Model dependent
        yt = self.get_yt(time)
        return -self.cot_angle * yt + self.csc_angle * (
            self.iretreat_model_spline(time) - self.iretreat_model_spline(0)
        )

    def get_trajectory(self):
        x = self.get_xt(self.ins_times)
        y = self.get_yt(self.ins_times)
        return x, y

    def get_nearest_points(self):
        x = self.get_xt(self.ins_times)
        y = self.get_yt(self.ins_times)
        xd = self.xdata
        yd = self.ydata
        xn = np.zeros_like(xd)
        yn = np.zeros_like(yd)
        for i in range(len(xd)):
            ind = np.argmin((x - xd[i]) ** 2 + (y - yd[i]) ** 2)
            xn[i] = x[ind]
            yn[i] = y[ind]
        return xn, yn

    def lnlikelihood(self):
        xd = self.xdata
        yd = self.ydata
        xn, yn = self.get_nearest_points()
        # Variance in meters in both directions
        xvar, yvar = (self.errorbar * self.meters_per_pixel) ** 2
        chi2 = (xd - xn) ** 2 / xvar + (yd - yn) ** 2 / yvar
        return -0.5 * chi2.sum() - 0.5 * self.Ndata * np.log(xvar * yvar)
