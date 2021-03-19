from unittest import TestCase

import numpy as np

from mars_troughs import Trough


class TroughTest(TestCase):
    def test_smoke(self):
        test_acc_params = [1e-6, 1e-11]
        acc_model_number = 1
        test_lag_params = [1, 1e-7]
        lag_model_number = 1
        errorbar = 100.0
        tr = Trough(
            test_acc_params,
            test_lag_params,
            acc_model_number,
            lag_model_number,
            errorbar,
        )
        assert tr is not None

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
