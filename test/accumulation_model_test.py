from unittest import TestCase

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from mars_troughs.accumulation_model import (
    LinearInsolation,
    LinearObliquity,
    QuadraticInsolation,
)


class LinearAccumulationTest(TestCase):
    def test_smoke(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = LinearInsolation(times, insolation)
        assert model is not None
        assert isinstance(model.constant, float)
        assert isinstance(model.slope, float)

    def test_parameter_names(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = LinearInsolation(times, insolation)
        assert model.parameter_names == ["constant", "slope"]

    def test_constant(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model1 = LinearInsolation(times, insolation)
        model2 = LinearInsolation(times, insolation, 2.0, 1e-7)
        assert model1.constant != model2.constant
        assert model1.slope != model2.slope
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearInsolation(time, insolation, inter, slope)
            assert model.constant == inter
            assert model.slope == slope
            accums = model.get_accumulation_at_t(time)
            assert (accums == inter + slope * model._var_data_spline(time)).all()

    def test_get_yt(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearInsolation(time, insolation, inter, slope)
            yt = model.get_yt(time)
            assert (
                yt
                == -(
                    inter * time
                    + (
                        slope
                        * (
                            model._int_var_data_spline(time)
                            - model._int_var_data_spline(0)
                        )
                    )
                )
            ).all()

    def test_get_xt(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        spline = np.linspace(0, 100, 1000)
        spline = IUS(time, insolation)
        csc_angle = np.radians(np.linspace(0, 360, 1000))
        cot_angle = np.radians(np.linspace(0, 360, 1000))
        constant = 1.0
        slope = 1e-6
        model = LinearInsolation(time, insolation, constant, slope)
        xt = model.get_xt(time, spline, cot_angle, csc_angle)
        assert (xt != np.nan).all()
        assert (xt != np.inf).all()
        assert np.size(xt) == np.size(time)


class QuadraticAccumulationTest(TestCase):
    def test_smoke(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = QuadraticInsolation(times, insolation)
        assert model is not None
        assert isinstance(model.constant, float)
        assert isinstance(model.slope, float)
        assert isinstance(model.quad, float)

    def test_parameter_names(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = QuadraticInsolation(times, insolation)
        assert model.parameter_names == [
            "constant",
            "slope",
            "quad",
        ]

    def test_constant(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model1 = QuadraticInsolation(times, insolation)
        model2 = QuadraticInsolation(times, insolation, 2.0, 1e-7, 1e-9)
        assert model1.constant != model2.constant
        assert model1.slope != model2.slope
        assert model1.quad != model2.quad
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope, quad in zip(
            [1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-6], [1e-8, 2e-8, 3e-8]
        ):
            model = QuadraticInsolation(time, insolation, inter, slope, quad)
            assert model.constant == inter
            assert model.slope == slope
            assert model.quad == quad
            accums = model.get_accumulation_at_t(time)
            truth = (
                inter
                + slope * model._var_data_spline(time)
                + quad * (model._var_data_spline(time)) ** 2
            )
            np.testing.assert_allclose(accums, truth)

    def test_get_yt(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for constant, slope, quad in zip(
            [1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-6], [1e-8, 2e-8, 3e-8]
        ):
            model = QuadraticInsolation(time, insolation, constant, slope, quad)
            yt = model.get_yt(time)
            assert (
                yt
                == -(
                    constant * time
                    + (
                        slope
                        * (
                            model._int_var_data_spline(time)
                            - model._int_var_data_spline(0)
                        )
                        + quad
                        * (
                            model._int_var2_data_spline(time)
                            - model._int_var2_data_spline(0)
                        )
                    )
                )
            ).all()

    def test_get_xt(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        spline = np.linspace(0, 100, 1000)
        spline = IUS(time, insolation)
        csc_angle = np.radians(np.linspace(0, 360, 1000))
        cot_angle = np.radians(np.linspace(0, 360, 1000))
        constant = 1.0
        slope = 1e-6
        quad = 1e-8
        model = QuadraticInsolation(time, insolation, constant, slope, quad)
        xt = model.get_xt(time, spline, cot_angle, csc_angle)
        assert (xt != np.nan).all()
        assert (xt != np.inf).all()
        assert np.size(xt) == np.size(time)


class ObliquityLinearAccumulationTest(TestCase):
    def test_smoke(self):
        obliquity = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = LinearObliquity(times, obliquity)
        assert model is not None
        assert isinstance(model.constant, float)
        assert isinstance(model.slope, float)

    def test_parameter_names(self):
        obliquity = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = LinearObliquity(times, obliquity)
        assert model.parameter_names == ["constant", "slope"]

    def test_constant(self):
        obliquity = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model1 = LinearObliquity(times, obliquity)
        model2 = LinearObliquity(times, obliquity, 2.0, 1e-7)
        assert model1.constant != model2.constant
        assert model1.slope != model2.slope
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        obliquity = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearObliquity(time, obliquity, inter, slope)
            assert model.constant == inter
            assert model.slope == slope
            accums = model.get_accumulation_at_t(time)
            assert (accums == inter + slope * model._var_data_spline(time)).all()

    def test_get_yt(self):
        time = np.linspace(0, 1e6, 1000)
        obliquity = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearObliquity(time, obliquity, inter, slope)
            yt = model.get_yt(time)
            assert (
                yt
                == -(
                    inter * time
                    + (
                        slope
                        * (
                            model._int_var_data_spline(time)
                            - model._int_var_data_spline(0)
                        )
                    )
                )
            ).all()

    def test_get_xt(self):
        time = np.linspace(0, 1e6, 1000)
        obliquity = np.sin(np.radians(np.linspace(0, 360, 1000)))
        spline = np.linspace(0, 100, 1000)
        spline = IUS(time, obliquity)
        csc_angle = np.radians(np.linspace(0, 360, 1000))
        cot_angle = np.radians(np.linspace(0, 360, 1000))
        constant = 1.0
        slope = 1e-6
        model = LinearObliquity(time, obliquity, constant, slope)
        xt = model.get_xt(time, spline, cot_angle, csc_angle)
        assert (xt != np.nan).all()
        assert (xt != np.inf).all()
        assert np.size(xt) == np.size(time)
