import tritonclient.grpc as grpc_client
from tritonclient.utils import np_to_triton_dtype
from tritonclient.grpc import service_pb2
from tritonclient.grpc.service_pb2 import model__config__pb2 as model_config_pb2

__version__ = "0.1.0"


def raise_error(msg):
    raise Exception(msg)


def model_config_from_model(name, url="localhost:8001"):
    client = grpc_client.InferenceServerClient(url)
    return client.get_model_config(name).config


def input_from_config(config):
    return config.input


def output_from_config(config):
    return config.output


def input_from_model(name, url="localhost:8001"):
    return input_from_config(model_config_from_model(name, url))


def output_from_model(name, url="localhost:8001"):
    return output_from_config(model_config_from_model(name, url))


# shapes and dtypes are checked in InferInput, when set_data_from_numpy() is invoked
def function_from_model(
    model_name, preprocessing=None, postprocessing=None, hostname="localhost",
    port=8001
):
    url = f"{hostname}:{port}"
    def _wrapper(*args, **kwargs):
        assert (len(args) == 0) ^ (
            len(kwargs) == 0
        ), "You may provide inputs in the form of either args or kwargs, but not both"
        actual_args = args if len(args) > 0 else kwargs

        client = grpc_client.InferenceServerClient(url)
        config = client.get_model_config(model_name).config
        model_input = input_from_config(config)
        model_output = output_from_config(config)
        inputs = []

        no_expected = len(model_input)
        no_provided = len(actual_args)

        if no_expected != no_provided:
            raise_error(f"the number of provided arguments ({no_expected}) and inputs ({no_provided}) must match")

        pre = preprocessing(actual_args) if preprocessing else actual_args
        
        if len(args) > 0:   
            for (a, b) in zip(pre, model_input):
                inputs.append(
                    grpc_client.InferInput(b.name, a.shape, np_to_triton_dtype(a.dtype))
                )
            for (a, b) in zip(pre, inputs):
                b.set_data_from_numpy(a)
        else:
            for b in model_input:
                a = pre[b.name]
                inputs.append(
                    grpc_client.InferInput(b.name, a.shape, np_to_triton_dtype(a.dtype))
                )
                inputs[-1].set_data_from_numpy(a)
        outputs = [grpc_client.InferRequestedOutput(o.name) for o in model_output]
        response = client.infer(model_name, inputs, model_version="1", outputs=outputs)
        result = [response.as_numpy(o.name) for o in model_output]
        if config.backend == "onnxruntime":
            result.reverse()
        if len(model_output) != len(result):
            raise_error("the number of result outputs and model outputs must match")
        post = postprocessing(result) if postprocessing else result
        return post

    return _wrapper
