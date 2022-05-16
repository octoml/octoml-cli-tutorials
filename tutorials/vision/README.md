# Vision Model

This is an example computer vision model for use with `octoml` cli.

## Note on Python3.10

The `onnxruntime` library is a dependency on these tutorials. It currently does not support python `>3.9`. For users locked into `3.10` you can run this vision tutorial by following the below steps. You can also run only `octoml` natively and use an `ubuntu:20.04` docker container to run the inferencing utility, `run.py`.

* Edit `requirements.txt` and remove `onnxruntime`
* Edit `setup.py` and remove the line that begins with `${PYTHON} -m transformers.onnx`
* Run `setup.py`
* Use this example only
