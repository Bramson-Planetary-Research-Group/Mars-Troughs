from unittest import TestCase

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

from mars_troughs import (
    LinearInsolationAccumulation,
    QuadraticInsolationAccumulation,
    ObliquityLinearAccumulation,
)


class LinearAccumulationTest(TestCase):
    def test_smoke(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = LinearInsolationAccumulation(times, insolation)
        assert model is not None
        assert isinstance(model.intercept, float)
        assert isinstance(model.slope, float)

    def test_parameter_names(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = LinearInsolationAccumulation(times, insolation)
        assert model.parameter_names == ["intercept", "slope"]

    def test_constant(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model1 = LinearInsolationAccumulation(times, insolation)
        model2 = LinearInsolationAccumulation(times, insolation, 2.0, 1e-7)
        assert model1.intercept != model2.intercept
        assert model1.slope != model2.slope
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearInsolationAccumulation(time, insolation, inter, slope)
            assert model.intercept == inter
            assert model.slope == slope
            accums = model.get_accumulation_at_t(time)
            assert (accums == inter + slope * model._var_data_spline(time)).all()

    def test_get_yt(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = LinearInsolationAccumulation(time, insolation, inter, slope)
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
        intercept = 1.0
        slope = 1e-6
        model = LinearInsolationAccumulation(time, insolation, intercept, slope)
        xt = model.get_xt(time, spline, cot_angle, csc_angle)
        assert (xt != np.nan).all()
        assert (xt != np.inf).all()
        assert np.size(xt) == np.size(time)


class QuadraticAccumulationTest(TestCase):
    def test_smoke(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = QuadraticInsolationAccumulation(times, insolation)
        assert model is not None
        assert isinstance(model.intercept, float)
        assert isinstance(model.linearCoeff, float)
        assert isinstance(model.quadCoeff, float)

    def test_parameter_names(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = QuadraticInsolationAccumulation(times, insolation)
        assert model.parameter_names == ["intercept", "linearCoeff", "quadCoeff"]

    def test_constant(self):
        insolation = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model1 = QuadraticInsolationAccumulation(times, insolation)
        model2 = QuadraticInsolationAccumulation(
            times, insolation, 2.0, 1e-7, 1e-9
        )
        assert model1.intercept != model2.intercept
        assert model1.linearCoeff != model2.linearCoeff
        assert model1.quadCoeff != model2.quadCoeff
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, linearCoeff, quadCoeff in zip(
            [1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-6], [1e-8, 2e-8, 3e-8]
        ):
            model = QuadraticInsolationAccumulation(
                time, insolation, inter, linearCoeff, quadCoeff
            )
            assert model.intercept == inter
            assert model.linearCoeff == linearCoeff
            assert model.quadCoeff == quadCoeff
            accums = model.get_accumulation_at_t(time)
            assert (
                accums
                == inter
                + linearCoeff * model._var_data_spline(time)
                + quadCoeff * (model._var_data_spline(time)) ** 2
            ).all()

    def test_get_yt(self):
        time = np.linspace(0, 1e6, 1000)
        insolation = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for intercept, linearCoeff, quadCoeff in zip(
            [1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-6], [1e-8, 2e-8, 3e-8]
        ):
            model = QuadraticInsolationAccumulation(
                time, insolation, intercept, linearCoeff, quadCoeff
            )
            yt = model.get_yt(time)
            assert (
                yt
                == -(
                    intercept * time
                    + (
                        linearCoeff
                        * (
                            model._int_var_data_spline(time)
                            - model._int_var_data_spline(0)
                        )
                        + quadCoeff
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
        intercept = 1.0
        linearCoeff = 1e-6
        quadCoeff = 1e-8
        model = QuadraticInsolationAccumulation(
            time, insolation, intercept, linearCoeff, quadCoeff
        )
        xt = model.get_xt(time, spline, cot_angle, csc_angle)
        assert (xt != np.nan).all()
        assert (xt != np.inf).all()
        assert np.size(xt) == np.size(time)

class ObliquityLinearAccumulationTest(TestCase):
    def test_smoke(self):
        obliquity = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = ObliquityLinearAccumulation(times, obliquity)
        assert model is not None
        assert isinstance(model.intercept, float)
        assert isinstance(model.slope, float)

    def test_parameter_names(self):
        obliquity = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model = ObliquityLinearAccumulation(times, obliquity)
        assert model.parameter_names == ["intercept", "slope"]

    def test_constant(self):
        obliquity = np.sin(np.radians(np.linspace(0, 360, 100)))
        times = np.linspace(0, 100, 100)
        model1 = ObliquityLinearAccumulation(times, obliquity)
        model2 = ObliquityLinearAccumulation(times, obliquity, 2.0, 1e-7)
        assert model1.intercept != model2.intercept
        assert model1.slope != model2.slope
        assert set(model1.parameters.keys()) == set(model2.parameters.keys())
        assert model1.parameters != model2.parameters

    def test_lag(self):
        time = np.linspace(0, 1e6, 1000)
        obliquity = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = ObliquityLinearAccumulation(time, obliquity, inter, slope)
            assert model.intercept == inter
            assert model.slope == slope
            accums = model.get_accumulation_at_t(time)
            assert (accums == inter + slope * model._var_data_spline(time)).all()

    def test_get_yt(self):
        time = np.linspace(0, 1e6, 1000)
        obliquity = np.sin(np.radians(np.linspace(0, 360, 1000)))
        for inter, slope in zip([1.0, 2.0, 3.0], [1e-6, 2e-6, 3e-5]):
            model = ObliquityLinearAccumulation(time, obliquity, inter, slope)
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
        intercept = 1.0
        slope = 1e-6
        model = ObliquityLinearAccumulation(time, obliquity, intercept, slope)
        xt = model.get_xt(time, spline, cot_angle, csc_angle)
        assert (xt != np.nan).all()
        assert (xt != np.inf).all()
        assert np.size(xt) == np.size(time)