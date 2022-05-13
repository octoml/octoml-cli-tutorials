import subprocess
import os
import time

from transformers.models.auto import AutoTokenizer
from transformers.onnx.features import FeaturesManager
from transformers.onnx.convert import export
from pathlib import Path


def get_models_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


def run_cmd(cmd):
    cmd = cmd if isinstance(cmd, list) else [cmd]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def retry_cmd(cmd, times):
    cmd = cmd if isinstance(cmd, list) else [cmd]
    for i in range(times):
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode == 0:
            break
        print("retrying {} after {} seconds".format(cmd, i + 1))
        time.sleep(1)


def pull_triton_image(image):
    cmd = ["docker", "pull", image]
    run_cmd(cmd)


def start_triton_container(image, name):
    models_path = get_models_path()
    mount_opt = "-v{}:/models".format(models_path)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p8000:8000",
        "-p8001:8001",
        "-p8002:8002",
        mount_opt,
        "--shm-size=1g",
        image,
        "tritonserver",
        "--model-repository=/models",
        "--strict-model-config=false",
    ]
    run_cmd(cmd)


def wait_triton_container(name):
    cmd = ["curl", "-v", "localhost:8000/v2/health/ready"]
    retry_cmd(cmd, 30)


def end_triton_container(name):
    cmd = ["docker", "rm", "-f", name]
    run_cmd(cmd)


def export_hf_model_to_onnx(model_name):
    model_path = os.path.join(get_models_path(), model_name, "1")
    if not os.path.exists(os.path.join(model_path, "model.onnx")):
        features = "default"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = FeaturesManager.get_model_from_feature(features, model_name)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model, feature=features
        )
        onnx_config = model_onnx_config(model.config)
        opset = onnx_config.default_onnx_opset
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, ("model.onnx"))
        export(tokenizer, model, onnx_config, opset, Path(model_path))


def pytest_sessionstart(session):
    export_hf_model_to_onnx("gpt2")
    container = "ariel_test"
    image = "nvcr.io/nvidia/tritonserver:22.01-py3"
    pull_triton_image(image)
    start_triton_container(image, container)
    wait_triton_container("foo")


def pytest_sessionfinish(session, exitstatus):
    container = "ariel_test"
    end_triton_container(container)
