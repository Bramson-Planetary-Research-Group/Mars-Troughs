from unittest import TestCase

import numpy as np

from mars_troughs import Trough


class TroughTest(TestCase):
    def setUp(self):
        self.acc_params = [1e-6, 1e-11]
        self.acc_model_number = 1
        self.lag_params = [1, 1e-7]
        self.lag_model_number = 1
        self.errorbar = 100.0

    def get_trough_object(self, **kwargs):
        return Trough(
            self.acc_params,
            self.lag_params,
            self.acc_model_number,
            self.lag_model_number,
            self.errorbar,
            **kwargs,
        )

    def test_smoke(self):
        tr = self.get_trough_object()
        assert tr is not None

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
