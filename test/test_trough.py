from unittest import TestCase

import numpy as np

from mars_troughs import Trough


class TroughTest(TestCase):
    def setUp(self):
        self.acc_params = [1e-6, 1e-11, 1e-11]
        self.acc_model_name = "quadratic"
        self.lag_params = [1, 1e-7]
        self.lag_model_name = "linear"
        self.errorbar = 100.0

    def get_trough_object(self, **kwargs):
        return Trough(
            self.acc_model_name,
            self.lag_model_name,
            self.acc_params,
            self.lag_params,
            self.errorbar,
            **kwargs,
        )

    def test_smoke(self):
        tr = self.get_trough_object()
        assert tr is not None

    def test_set_model(self):
        self.acc_params = [1e-6, 1e-11]
        self.acc_model_name = "linear"
        self.lag_params = [1]
        self.lag_model_name = "constant"
        tr = self.get_trough_object()
        # Trajectory predictions
        x1, y1 = tr.get_trajectory()
        # Come up with new parameters
        acc_params = {"intercept": 1.0, "slope": 2.0}
        lag_params = {"constant": 33.0}
        errorbar = 200.0
        tr.set_model(acc_params, lag_params, errorbar)
        assert tr.accuModel.parameters == acc_params
        assert tr.lagModel.parameters == lag_params
        assert tr.errorbar == errorbar
        # Make sure the trajectory predictions changed
        # meaning the splines changed
        # Note - first point is 0 so it doesn't change
        x2, y2 = tr.get_trajectory()
        assert (x1 != x2)[1:].all()
        assert (y1 != y2)[1:].all()

    def test_get_trajectory(self):
        tr = self.get_trough_object()
        x, y = tr.get_trajectory()
        assert len(x) == len(y)
        assert len(x) == len(tr.times)

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
