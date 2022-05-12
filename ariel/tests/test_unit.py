from ariel import __version__
from ariel import input_from_model, output_from_model, function_from_model
import numpy as np
from ariel import model_config_pb2


def test_version():
    assert __version__ == "0.1.0"


def test_input_names():
    inputs = input_from_model("py_model")
    res = [i.name for i in inputs]
    exp = ["INPUT0", "INPUT1"]
    assert res, exp


def test_input_types():
    dtype = model_config_pb2.DataType.TYPE_FP32
    inputs = input_from_model("py_model")
    res = [i.data_type for i in inputs]
    exp = [dtype, dtype]
    assert res, exp


def test_input_dims():
    inputs = input_from_model("py_model")
    res = [i.dims for i in inputs]
    exp = [[4], [4]]
    assert res, exp


def test_output_names():
    outputs = output_from_model("py_model")
    res = [o.name for o in outputs]
    exp = ["OUTPUT0", "OUTPUT1"]
    assert res, exp


def test_output_types():
    outputs = output_from_model("py_model")
    res = [o.data_type for o in outputs]
    exp = [11, 11]
    assert res, exp


def test_output_dims():
    outputs = output_from_model("py_model")
    res = [o.dims for o in outputs]
    exp = [[4], [4]]
    assert res, exp
