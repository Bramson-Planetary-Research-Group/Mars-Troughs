"""
The trough model.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from mars_troughs.model import Model
from typing import Any, Dict, List, Optional


class Trough():
    """
    This object models trough migration patterns (TMPs). It is composed of
    a model for the accumulation of ice on the surface of the trough, accessible
    as the :attr:`accuModel` attribute, as well as a model for the lag
    that builds up over time, accesible as the :attr:`lagModel` attribute.

    Args:
      acc_model (Union[str, Model]): name of the accumulation model
        (linear, quadratic, etc) or a custom model
      lag_model_name (Union[str, Model]): name of the lag(t) model (constant,
        linear, etc) or a custom model
      acc_params (List[float]): model parameters for accumulation
      lag_params (List[float]): model parameters for lag(t)
      errorbar (float, optional): errorbar of the datapoints in pixels; default=1
      angle (float, optional): south-facing slope angle in degrees. Default is 2.9.
      insolation_path (Union[str, Path], optional): path to the file with
        insolation data.
    """

    def __init__(
        self,
        acc_model: Union[str, Model],
        lag_model: Union[str, Model],
        ret_data_spline,
        angle: float = 2.9,
    ):
        """Constructor for the trough object.
        Args:
          acc_model (Model): accumulation submodel
          lag_model (Model): lag submodel
          errorbar (float, optional): errorbar of the datapoints in pixels; default=1
          angle (float, optional): south-facing slope angle in degrees. Default is 2.9.
        """

        self.accuModel = acc_model
        self.lagModel = lag_model
        self.ret_data_spline=ret_data_spline
        self.angle = angle
        self.meters_per_pixel = np.array([475.0, 20.0])  # meters per pixel
        self.errorbar = 1 #error is 1 pixel always

        # Set submodels
        self.set_submodels([self.accuModel, self.lagModel])

        # Calculate the model of retreat of ice per time
        self.lag_at_t=self.lagModel.get_lag_at_t(self.accuModel._times)
        self.retreat_model_t = self.ret_data_spline.ev(self.lag_at_t, 
                                                       self.accuModel._times)

        # Compute the Retreat(time) spline
        self.retreat_model_t_spline = IUS(self.accuModel._times, 
                                          self.retreat_model_t)

    def set_model(
        self,
        all_parameters: Dict[str, float],
    ) -> None:
        """
        Updates trough model with new accumulation and lag parameters.
        Then updates all splines.

        Args:
            all_parameter (Dict[str, float]): new parameters to the models
        """
        self.all_parameters = all_parameters

        # Update the model of retreat of ice per time
        self.lag_at_t=self.lagModel.get_lag_at_t(self.accuModel._times)
        self.retreat_model_t = self.ret_data_spline.ev(self.lag_at_t, 
                                                       self.accuModel._times)

        # Update the Retreat(time) spline
        self.retreat_model_t_spline = IUS(self.accuModel._times, 
                                          self.retreat_model_t)
        return

    def get_trajectory(
        self, times: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtains the x and y coordinates (in m) of the trough model as a
        function of time.

        Args:
            times (Optional[np.ndarray]): if ``None``, default to the
                times of the observed solar insolation.

        Output:
            x and y coordinates (tuple) of size 2 x len(times) (in m).
        """

        y = self.accuModel.get_yt(times)
        x = self.accuModel.get_xt(
            times,
            self.retreat_model_t_spline.antiderivative(),
            self.cot_angle,
            self.csc_angle,
        )

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
        times: Optional[np.ndarray] = None,
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
        x_model, y_model = self.get_trajectory(times)
        x_out = np.zeros_like(x_data)
        y_out = np.zeros_like(y_data)
        time_out = np.zeros_like(y_data)
        for i, (xdi, ydi) in enumerate(zip(x_data, y_data)):
            dist = dist_func(x_model, xdi, y_model, ydi)
            ind = np.argmin(dist)
            x_out[i] = x_model[ind]
            y_out[i] = y_model[ind]
            time_out[i] = times[ind]
        return x_out, y_out, time_out

    def lnlikelihood(
            self, 
            x_data: np.ndarray, 
            y_data: np.ndarray,
            times: Optional[np.ndarray] = None,) -> float:
        """
        Calculates the log-likelihood of the data given the model.
        Note that this is the natural log (ln).

        Args:
            x_data (np.ndarray): x-coordinates of the trough path
            y_data (np.ndarray): y-coordinates of the trough path

        Output:
            log-likelihood value (float)
        """
        self.xnear, self.ynear, self.timesxy = self.get_nearest_points(
                                                      x_data, y_data,times)
        # Variance in meters in both directions
        xvar, yvar = (self.errorbar * self.meters_per_pixel) ** 2
        chi2 = (x_data - self.xnear) ** 2 / xvar + (y_data - 
                                                      self.ynear) ** 2 / yvar
        return (-0.5 * chi2.sum() 
                -0.5 * len(x_data) * np.log(2*np.pi) 
                -0.5 * np.log(xvar * yvar))
    
    def set_submodels(self, sub_models: Optional[List["Model"]] = None) -> None:
        # Add sub_models as an attribute
        self.sub_models: List[Model] = sub_models or []

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
        
    @property
    def all_parameter_names(self) -> List[str]:
        """
        Names of the parameters of all sub-models.

        Returns:
          names of parameters of all sub-models
        """
        return self.all_parameters.keys()

    @property
    def all_parameters(self) -> Dict[str, Any]:
        """
        The parameters for this model and all sub-models.

        Returns:
          key-value pairs for this model and all sub-models
        """
        # Find parameters of sub_models
        sub_pars = {}
        for i, sub_model in enumerate(self.sub_models):
            # If the sub model doesn't have a prefix then prepend a number
            sub_prefix = sub_model.prefix or str(i) + "_"
            # Attach the prefix of submodels to their keys
            for key, value in sub_model.parameters.items():
                sub_pars[sub_prefix + key] = value

        # Bundle up our parameters and all sub model's parameters
        return {**sub_pars}

    @all_parameters.setter
    def all_parameters(self, params: Dict[str, Any]) -> None:
        """
        Setter for all sub-models parameters.

        Args:
          params (Dict[str, Any]): new parameters
        """
        # Update sub model's parameters
        for i, sub_model in enumerate(self.sub_models):
            # Pull out only the parameters associated with the sub model,
            # based on its prefix (or the default assigned prefix)
            # Set the sub model's `all_parameters` to be this parameter subset
            sub_prefix = sub_model.prefix or str(i) + "_"
            sub_params = {
                k[len(sub_prefix) :]: v
                for k, v in params.items()
                if k.startswith(sub_prefix)
            }
            sub_model.parameters = sub_params
        return
