#!/bin/bash
set -eux

# Get setup dir
SETUP_DIR=$(dirname $0)

PYTHON=python
# Compat for python3
if ! command -v python &> /dev/null
then
    PYTHON=python3
fi

# Install pip requirements
pip install -r requirements.txt


# Get the bert model for the question answering demo
${PYTHON} -m transformers.onnx --model=bert-large-uncased-whole-word-masking-finetuned-squad --feature=question-answering qa

# Get the gpt2 model for the generation model
curl -fsSL -o generation/gpt2-lm-head-10.onnx https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx

# Install ariel lib
pip install ${SETUP_DIR}/../ariel

# Check for docker installation
if ! command -v docker &> /dev/null
then
    echo "ERROR: `docker` not found. Please install docker: https://docs.docker.com/engine/install/"
    exit 1
fi
