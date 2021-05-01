from unittest import TestCase

from mars_troughs import Model


class DummyModel(Model):
    a: float = 1.0
    b: float = 10.0
    c: float = 20.0

    @property
    def parameter_names(self):
        return ["a", "b", "c"]


class OtherModel(Model):
    x: float = 3.0
    y: float = 4.0

    @property
    def parameter_names(self):
        return ["x", "y"]


class TestModel(TestCase):
    def test_smoke(self):
        model = DummyModel()
        assert isinstance(model, Model)
        assert model.parameter_names == ["a", "b", "c"]
        assert model.sub_models == []
        assert model.all_parameter_names == ["a", "b", "c"]
        assert model.parameters == {"a": 1.0, "b": 10.0, "c": 20.0}
        # test setting one parameter
        model.a = 30.0
        assert model.parameters == {"a": 30.0, "b": 10.0, "c": 20.0}
        # Test setting all parameters
        model.parameters = {"a": -1.0, "b": -10.0, "c": -20.0}
        assert model.parameters == {"a": -1.0, "b": -10.0, "c": -20.0}

    def test_with_submodel(self):
        sub = OtherModel()
        model = DummyModel(sub_models=[sub])
        assert model.parameter_names == ["a", "b", "c"]
        assert model.sub_models == [sub]
        assert model.parameters == {"a": 1.0, "b": 10.0, "c": 20.0}
        assert model.all_parameters == {
            "a": 1.0,
            "b": 10.0,
            "c": 20.0,
            "x": 3.0,
            "y": 4.0,
        }
        model.parameters = {"a": -1.0, "b": -10.0, "c": -20.0}
        assert model.all_parameters == {
            "a": -1.0,
            "b": -10.0,
            "c": -20.0,
            "x": 3.0,
            "y": 4.0,
        }
        model.all_parameters = {
            "a": -2.0,
            "b": -20.0,
            "c": -40.0,
            "x": 33.0,
            "y": 44.0,
        }
        assert model.all_parameters == {
            "a": -2.0,
            "b": -20.0,
            "c": -40.0,
            "x": 33.0,
            "y": 44.0,
        }
        assert model.parameters == {"a": -2.0, "b": -20.0, "c": -40.0}
        assert sub.parameters == {"x": 33.0, "y": 44.0}
