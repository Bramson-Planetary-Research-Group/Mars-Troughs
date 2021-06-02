from unittest import TestCase
import numpy as np
from mars_troughs import Trough, Model, DATAPATHS
from typing import List
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
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

        return self.coeff * self._inv_ins
    
    def get_yt(self, time: np.ndarray):

        return -self.coeff * ( self._int_inv_ins_data_spline(time) - 
                               self._int_inv_ins_data_spline(0) )
    
    def get_xt(
        self,
        time: np.ndarray,
        int_retreat_model_t_spline: np.ndarray,
        cot_angle,
        csc_angle,
    ):

        yt = self.get_yt(time)

        return -cot_angle * yt + csc_angle * (
            int_retreat_model_t_spline(time) - int_retreat_model_t_spline(0)
        )
                              
    @property
    def parameter_names(self) -> List[str]:
        return ["coeff"]

class CustomLagModel(Model):
    """
    Custom lag model
    """
    def __init__(
        self,
        intercept: float=1e-6,
        linearCoeff: float = 1e-6,
        quadCoeff: float = 1e-6,
        ):
        
        self.intercept=intercept
        self.linearCoeff=linearCoeff
        self.quadCoeff=quadCoeff
        

    def get_lag_at_t(self, time: np.ndarray) -> np.ndarray:

        return self.intercept + self.linearCoeff*time + self.quadCoeff*time**2

    @property
    def parameter_names(self) -> List[str]:
        return ["intercept", "linearCoeff","quadCoeff"]

class TroughTestCustom(TestCase):
    def setUp(self):
        self.acc_params = None
        self.acc_model = CustomAccuModel(1e-6)
        self.lag_params = None
        self.lag_model = CustomLagModel(1e-6,1e-6,1e-6)
        self.errorbar = 100.0

    def get_trough_object(self, **kwargs):
        return Trough(
            self.acc_model,
            self.lag_model,
            self.acc_params,
            self.lag_params,
            self.errorbar,
            **kwargs,
        )

    def test_smoke(self):
        tr = self.get_trough_object()
        assert tr is not None

    def test_set_model(self):
        tr = self.get_trough_object()
        # Come up with new parameters
        acc_params = {"coeff": 1e-6}
        lag_params = {"intercept":1e-7,"linearCoeff": 2e-6, "quadCoeff":2e-6}
        errorbar = 200.0
        tr.set_model(acc_params, lag_params, errorbar)
        assert tr.accuModel.parameters == acc_params
        assert tr.lagModel.parameters == lag_params
        assert tr.errorbar == errorbar

    def test_get_trajectory(self):
        tr = self.get_trough_object()
        x, y = tr.get_trajectory()
        assert len(x) == len(y)
        assert len(x) == len(tr.ins_times)

    def test_get_trajectory_input_times(self):
        tr = self.get_trough_object()
        times = np.linspace(0, 100)
        x, y = tr.get_trajectory(times)
        assert len(x) == len(y)
        assert len(x) == len(times)

    def test_angle(self):
        tr1 = self.get_trough_object(angle=2.9)
        tr2 = self.get_trough_object(angle=3.9)
        assert tr1.angle != tr2.angle
        assert tr1.csc_angle != tr2.csc_angle
        assert tr1.cot_angle != tr2.cot_angle

    def test_get_nearest_points(self):
        tr = self.get_trough_object()
        # Junk data
        x = y = -np.arange(10) * 100
        xo, yo = tr.get_nearest_points(x, y)
        assert xo.shape == x.shape
        assert yo.shape == y.shape

    def test_lnlikelihood(self):
        tr = self.get_trough_object()
        # Junk data
        x = y = -np.arange(10) * 100
        LL = tr.lnlikelihood(x, y)
        assert LL < 0

    def test__L2_distance(self):
        N = 10
        for N in [10, 100, 357, 1000]:
            x1 = np.arange(N)
            x2 = N / 2.0 + 0.1
            y1 = np.arange(N) + 20
            y2 = N / 2.0 + 20.1
            dist = Trough._L2_distance(x1, x2, y1, y2)
            assert (dist == ((x1 - x2) ** 2 + (y1 - y2) ** 2)).all()
            assert np.argmin(dist) in [N // 2, N // 2 + 1]
            
    
                
