#!/bin/bash
set -eux

PYTHON=python
# Compat for python3
if ! command -v python &> /dev/null
then
    PYTHON=python3
fi

# Get the bert model for the question answering demo
${PYTHON} -m transformers.onnx --model=bert-large-uncased-whole-word-masking-finetuned-squad --feature=question-answering question_answering

# Get the gpt2 model for the generation model
curl -fsSL -o generation/gpt2-lm-head-10.onnx https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
