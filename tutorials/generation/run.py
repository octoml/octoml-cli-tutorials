import argparse
import torch
from typing import Union
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from onnxruntime import InferenceSession

# import triton_util from parent dir
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from triton_util import TritonRemoteModel  # noqa


MAX_SEQUENCE_LENGTH = 100
SAMPLE_TEXT = "Hello, I'm a language model,"


def onnx_model_wrapper():
    model = InferenceSession('model.onnx')

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        input_feed = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
        }
        logits = model.run(output_names=['logits'], input_feed=input_feed)
        return CausalLMOutputWithCrossAttentions(logits=torch.tensor(logits[0]))

    return forward


def triton_model_wrapper(server_url, model_name, protocol):
    model = TritonRemoteModel(server_url, model_name, protocol=protocol)

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        input_feed = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
        }
        logits = model(**input_feed)
        return CausalLMOutputWithCrossAttentions(logits=torch.tensor(logits[0]))

    return forward


def run_local():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_inputs = tokenizer(SAMPLE_TEXT, return_tensors='pt')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token_id)
    model.forward = onnx_model_wrapper()
    output = model.generate(**encoded_inputs)
    print(tokenizer.batch_decode(output))


def run_triton(port: Union[str, None], hostname: str, protocol: str):
    if port is None:
        port = '8000' if protocol == 'http' else '8001'
    server_url = f'{hostname}:{port}'

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_inputs = tokenizer(SAMPLE_TEXT, return_tensors='pt')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token_id)
    # TODO: Replace with a more efficient way
    # Currently way is inefficient as the remote model will be called inside a
    # loop of autoregressive decoding.
    model.forward = triton_model_wrapper(server_url, model_name='distilgpt2', protocol=protocol)
    output = model.generate(**encoded_inputs)
    print(tokenizer.batch_decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose mode to run inference in.')
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--triton", default=False, action="store_true")
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--port", default=None)
    parser.add_argument("--protocol", default="grpc", choices=["grpc", "http"])
    args = parser.parse_args()

    if args.local:
        run_local()

    if args.triton:
        run_triton(args.port, args.hostname, args.protocol)
