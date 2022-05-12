import argparse
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np
from ariel import function_from_model

# Note that MAX_SEQUENCE LENGTH must match the sequence lengths given in `octoml.toml`
MAX_SEQUENCE_LENGTH = 256
SAMPLE_QUESTION = "What are some example applications of BERT?"
SAMPLE_CONTEXT = """…BERT model can be finetuned with just one additional output layer
        to create state-of-the-art models for a wide range of tasks, such as
        question answering and language inference, without substantial
        task-specific architecture modifications."""


def tokenize_inputs(question, context):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    # ONNX Runtime expects NumPy arrays as input
    # Note that the input will be implicitly truncated to MAX_SEQUENCE_LENGTH
    encoded_input = tokenizer.encode_plus(
        question,
        context,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="np"
    )

    print("Question:", SAMPLE_QUESTION)
    return dict(encoded_input)


def interpret_output_logits(outputs, encoded_input):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    input_ids = encoded_input['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

    # Find the tokens with the highest `start` and `end` scores.
    start_scores, end_scores = outputs
    answer_start = np.argmax(start_scores)
    answer_end = np.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end+1])
    print("Answer:", answer)


def run_local():
    # Preprocess input question
    encoded_input = tokenize_inputs(SAMPLE_QUESTION, SAMPLE_CONTEXT)

    # Initialize the model
    session = InferenceSession('model.onnx')

    # Run inference
    outputs = session.run(output_names=["start_logits", "end_logits"], input_feed=encoded_input)

    # Interpret predictions
    interpret_output_logits(outputs, encoded_input)


def run_triton(port):
    # Preprocess input question
    encoded_input = tokenize_inputs(SAMPLE_QUESTION, SAMPLE_CONTEXT)

    # Initialize the model
    model = function_from_model("bert", port=port)

    # Run inference
    outputs = model(**encoded_input)

    # Interpret predictions
    interpret_output_logits(outputs, encoded_input)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Choose mode to run inference in.')
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--triton", default=False, action="store_true")
    parser.add_argument("--port", default=8001)
    args = parser.parse_args()

    if args.local:
        run_local()
    
    if args.triton:
        run_triton(args.port)
