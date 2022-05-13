from __future__ import annotations
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ariel import function_from_model
from onnxruntime import InferenceSession
import torch
import numpy as np

MAX_SEQUENCE_LENGTH = 100
SAMPLE_TEXT = "Hello, I'm a language model,"

def tokenize_inputs(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_inputs = tokenizer(text, return_tensors="pt")
    return encoded_inputs


def decode_outputs(outputs):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    decoded_output = tokenizer.batch_decode(outputs)
    return decoded_output


def generator_from_model(port=None):
    from transformers import GPT2LMHeadModel
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if port is None:
        session = InferenceSession("gpt2-lm-head-10.onnx")
    else:
        model_func = function_from_model("gpt2-lm-head-10", port=port)

    def onnx_eval(*args, **kwargs):
        input_ids = kwargs["input_ids"]
        onnx_inputs = {}
        # gpt2 onnx model expects an extra dimension for some reason
        onnx_inputs["input1"] = np.expand_dims(input_ids.numpy(), axis=0)
        if port is None:
            onnx_out_names = [x.name for x in session.get_outputs()]
            outputs = session.run(input_feed=onnx_inputs, output_names=onnx_out_names)
        else:
            outputs = model_func(**onnx_inputs)
        # unwrap the extra dimension
        logits = torch.tensor(outputs[0][0])
        res = CausalLMOutputWithCrossAttentions(logits=logits)
        return res

    model.forward = onnx_eval

    def _wrapper(encoded_inputs):
        return model.generate(
            encoded_inputs.input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=MAX_SEQUENCE_LENGTH,
        )

    return _wrapper

def run(port=None):
    # Preprocess input text
    encoded_inputs = tokenize_inputs(SAMPLE_TEXT)

    # Prepare model
    model = generator_from_model(port)
    
    # Generate output
    outputs = model(encoded_inputs)
    decoded_outputs = decode_outputs(outputs)
    print(decoded_outputs)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Choose mode to run inference in.')
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--triton", default=False, action="store_true")
    parser.add_argument("--port", default=8001)
    args = parser.parse_args()

    if args.local:
        run()
    
    if args.triton:
        run(args.port)
