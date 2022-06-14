"""Utility for interacting with models on Triton server

For more examples, please refer to `Triton documentation
<https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient>`__
and `examples<https://github.com/triton-inference-server/client/tree/main/src/python/examples>`__
"""

import numpy as np
from typing import Union, Tuple
from attrdict import AttrDict
try:
    from tritonclient.http import (
        InferenceServerClient as HttpClient,
        InferInput as HttpInferInput
    )
    _has_http_client = True
except ImportError:
    _has_http_client = False

try:
    from tritonclient.grpc import (
        InferenceServerClient as GrpcClient,
        InferInput as GrpcInferInput
    )
    _has_grpc_client = True
except ImportError:
    _has_grpc_client = False

if not (_has_grpc_client or _has_http_client):
    raise ImportError("Cannot import tritonclient, please run `pip install tritonclient[http,grpc]`")


class TritonRemoteModel:
    def __init__(self, url: str, model_name: str, model_version: str = "", protocol: str = "grpc"):
        """
        A wrapper over model on Tirton server to behave like a local model.
        After initializes, the model can take numpy array(s) as inputs and
        return numpy array(s) as outputs using the selected protocols.

        Args:
            url: Address of the triton server
            model_name: Name of the model
            model_version: Version of the model. Default empty will let the server pick.
            protocol: Choose over "GRPC" or "HTTP" as ways to communicate with the server.

        Notes:
            The triton inference server supports remote inferencing over both
            http and gRPC. HTTP is ubiquitous and we expect most users to be
            comfortable using the HTTP api and integrating it into their
            network architecture. gRPC is another remote inferencing protocol
            and it has much higher (network) performance and throughput than
            http.
        """
        if protocol == "grpc":
            assert _has_grpc_client, "Cannot import tritonclient.grpc. Please run `pip install tritonclient[grpc]`"
            InferenceServerClient = GrpcClient
            InferInput = GrpcInferInput
        else:
            assert _has_http_client, "Cannot import tritonclient.http. Please run `pip install tritonclient[http]`"
            InferenceServerClient = HttpClient
            InferInput = HttpInferInput

        self._client = InferenceServerClient(url)
        self._name = model_name
        self._version = model_version
        self._metadata = self._client.get_model_metadata(model_name, model_version)
        if protocol == 'http':
            self._metadata = AttrDict(self._metadata)
        self._infer_inputs = [InferInput(x.name, None, x.datatype) for x in self._metadata.inputs]
        self._protocol = protocol

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def inputs(self):
        return self._metadata.inputs

    @property
    def outputs(self):
        return self._metadata.outputs

    @property
    def backend(self):
        return self._metadata.platform

    @property
    def protocol(self) -> str:
        return self._protocol

    def __str__(self):
        input_sig = tuple(x.name for x in self.inputs)
        output_sig = tuple(x.name for x in self.outputs)
        return (f'Name: {self.name}[{self.version}]\n'
                f'Inputs: {input_sig}\n'
                f'Outputs: {output_sig}\n'
                f'Backend: {self.backend}')

    def _set_inputs(self, *args, **kwargs):
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError("Cannot specify args, and kwargs at the same time")
        if len(args) == 0 and len(kwargs) == 0:
            raise RuntimeError("At least one input needs to be specified. Got none.")

        if len(args) > 0:
            if len(args) != len(self._infer_inputs):
                raise RuntimeError("Expect {} inputs got {}".format(len(self._infer_inputs), len(args)))
            for placeholder, value in zip(self._infer_inputs, args):
                placeholder.set_shape(value.shape)
                placeholder.set_data_from_numpy(value)
        else:
            if len(kwargs) != len(self._infer_inputs):
                raise RuntimeError("Expect {} inputs got {}".format(len(self._infer_inputs), len(kwargs)))
            for placeholder in self._infer_inputs:
                value = kwargs[placeholder.name()]
                placeholder.set_shape(value.shape)
                placeholder.set_data_from_numpy(value)

    def __call__(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        self._set_inputs(*args, **kwargs)
        response = self._client.infer(
            model_name=self.name,
            inputs=self._infer_inputs,
            model_version=self.version)
        outputs = tuple(response.as_numpy(o.name) for o in self.outputs)
        return outputs
