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

    def test_all_parameter_names(self):
        tr = self.get_trough_object()
        assert "errorbar" in tr.all_parameter_names
        # 6 params based on setUp() -- 3 from acc
        # 2 from lag, 1 errorbar
        assert len(tr.all_parameter_names) == 6

    def test_set_model(self):
        # Test that we can update the trough object
        self.acc_params = [1e-6, 1e-11]
        self.acc_model_name = "linear"
        tr = self.get_trough_object()
        # Save a pointer to the old spline so we can compare
        old_spline = tr.retreat_model_t_spline
        # Also save the integrated retreat table
        old_model_retreat = tr.retreat_model_t
        # Update the model, mimicing how
        # we would pass new parameters in from our sampler
        new_params = {
            "acc_constant": 1,
            "acc_slope": 2,
            "lag_constant": 3,
            "lag_slope": 4,
            "errorbar": 5,
        }
        tr.set_model(new_params)
        assert tr.accuModel.parameters == {"constant": 1, "slope": 2}
        assert tr.lagModel.parameters == {"constant": 3, "slope": 4}
        assert tr.errorbar == 5
        assert tr.retreat_model_t_spline is not old_spline
        assert np.any(old_model_retreat != tr.retreat_model_t)

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

    def test_obliquity(self):
        self.acc_params = [1e-6, 1e-11]
        self.acc_model_name = "obliquity"
        self.lag_params = [1]
        self.lag_model_name = "constant"
        tr = self.get_trough_object()
        # Save a pointer to the old spline so we can compare
        old_spline = tr.retreat_model_t_spline
        # Also save the integrated retreat table
        old_model_retreat = tr.retreat_model_t
        # Come up with new parameters
        new_params = {
            "acc_constant": 1,
            "acc_slope": 2,
            "lag_constant": 3,
            "errorbar": 5,
        }
        tr.set_model(new_params)
        assert tr.accuModel.parameters == {"constant": 1, "slope": 2}
        assert tr.lagModel.parameters == {"constant": 3}
        assert tr.errorbar == 5
        assert tr.retreat_model_t_spline is not old_spline
        assert np.any(old_model_retreat != tr.retreat_model_t)
