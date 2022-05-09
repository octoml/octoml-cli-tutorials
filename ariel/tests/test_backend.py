from ariel import function_from_model
import onnxruntime as ort
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import os
import torch


def get_model_path(model_name, ext):
    model_file = "model.{}".format(ext)
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "models",
        model_name,
        "1",
        model_file,
    )


def preprocessing(args):
    result = []
    for a in args:
        result.append(a * 2)
    return result


def postprocessing(args):
    result = []
    for a in args:
        result.append(a - 2)
    return result


def test_py_model():
    n = 4
    m = 10
    a = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    b = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    f = function_from_model("py_model")
    res = f(a, b)
    exp = [a + b, a - b]
    for (a, b) in zip(exp, res):
        np.testing.assert_array_equal(a, b)


def test_py_model_with_preprocessing():
    n = 4
    m = 10
    a = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    b = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    f = function_from_model("py_model", preprocessing=preprocessing)
    res = f(a, b)
    pre = preprocessing([a, b])
    exp = [pre[0] + pre[1], pre[0] - pre[1]]
    for (a, b) in zip(exp, res):
        np.testing.assert_array_equal(a, b)


def test_py_model_with_postprocessing():
    n = 4
    m = 10
    a = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    b = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    f = function_from_model("py_model", postprocessing=postprocessing)
    res = f(a, b)
    exp = postprocessing([a + b, a - b])
    for (a, b) in zip(exp, res):
        np.testing.assert_array_equal(a, b)


def test_py_model_with_preprocessing_and_postprocessing():
    n = 4
    m = 10
    a = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    b = np.random.randint(m, size=n, dtype=np.int32).astype(np.float32)
    f = function_from_model(
        "py_model", preprocessing=preprocessing, postprocessing=postprocessing
    )
    res = f(a, b)
    pre = preprocessing([a, b])
    exp = postprocessing([pre[0] + pre[1], pre[0] - pre[1]])
    for (a, b) in zip(exp, res):
        np.testing.assert_array_equal(a, b)


def test_hf_transformer_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    remote_input = tokenizer(text, return_tensors="np")
    remote_input = [t for t in remote_input.values()]
    remote_input.reverse()
    remote_model = function_from_model(model_name)
    remote_output = remote_model(*remote_input)
    np.testing.assert_almost_equal(output[0].detach().numpy(), remote_output[0], 3)


def test_onnx_transformer_model():
    text = "hello world."
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input = tokenizer(text, return_tensors="np")
    ort_session = ort.InferenceSession(get_model_path(model_name, "onnx"))
    ort_output = ort_session.run(["last_hidden_state"], dict(input))
    remote_input = [t for t in input.values()]
    # reorder inputs
    remote_input.reverse()
    remote_model = function_from_model(model_name)
    remote_output = remote_model(*remote_input)
    np.testing.assert_almost_equal(ort_output, remote_output, 3)


def test_tensorflow_model():
    n = 16
    m = 10
    a = np.random.randint(m, size=n, dtype=np.int32)
    b = np.random.randint(m, size=n, dtype=np.int32)
    f = function_from_model("tf_model")
    res = f(a, b)
    exp = [a + b, a - b]
    for (a, b) in zip(exp, res):
        np.testing.assert_array_equal(a, b)


def test_torch_model():
    model_name = "torch_model"
    n = (3, 4)
    a = np.random.rand(*n).astype(np.float32)
    b = np.random.rand(*n).astype(np.float32)
    f = function_from_model(model_name)
    res = f(a, b)
    torch_model = torch.jit.load(get_model_path(model_name, "pt"))
    torch_model.eval()
    exp = torch_model(torch.from_numpy(a), torch.from_numpy(b))
    for (a, b) in zip(exp, res):
        np.testing.assert_array_equal(a, b)
