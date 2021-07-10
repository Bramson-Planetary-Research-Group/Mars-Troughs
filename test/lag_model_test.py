"""
Tests of the lag models.
"""

from unittest import TestCase

import numpy as np

from mars_troughs import ConstantLag, LinearLag


class ConstantLagTest(TestCase):
    def test_smoke(self):
        model = ConstantLag()
        assert model is not None
        assert isinstance(model.lag_constant, float)

    def test_parameter_names(self):
        model = ConstantLag()
        assert model.parameter_names == ["lag_constant"]

    def test_constant(self):
        model1 = ConstantLag()
        model2 = ConstantLag(constant=2.0)
        assert model1.lag_constant != model2.lag_constant
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        for const in [1.0, 2.0, 2.5]:
            model = ConstantLag(const)
            lags = model.get_lag_at_t(time)
            assert (lags == np.ones_like(time) * const).all()


class LinearLagTest(TestCase):
    def test_smoke(self):
        model = LinearLag()
        assert model is not None
        assert isinstance(model.lag_intercept, float)
        assert isinstance(model.lag_slope, float)

    def test_parameter_names(self):
        model = LinearLag()
        assert model.parameter_names == ["lag_intercept", "lag_slope"]

    def test_constant(self):
        model1 = LinearLag()
        model2 = LinearLag(2.0, 1e-7)
        assert model1.lag_intercept != model2.lag_intercept
        assert model1.lag_slope != model2.lag_slope
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearLag(inter, slope)
            assert model.lag_intercept == inter
            assert model.lag_slope == slope
            lags = model.get_lag_at_t(time)
            assert (lags == inter + slope * time).all()
