#!/bin/bash
set -eux

# Get setup dir
SETUP_DIR=$(dirname `readlink -f $0`)

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
curl -fsSL -o generation/gpt2-lm-head-10.onnx https://github.com/onnx/models/blob/main/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
# Install ariel lib
pip install ${SETUP_DIR}/../ariel

# Install aws cli if not already installed
if ! command -v aws &> /dev/null
then
    rm -f awscliv2.zip
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    rm -fr aws
    unzip awscliv2.zip
    sudo ./aws/install
fi


# Install kubectl if not already installed
if ! command -v kubectl &> /dev/null
then
    curl -fsSL -O "https://dl.k8s.io/release/v1.22.0/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
fi

# Install helm if not already installed
if ! command -v helm &> /dev/null
then
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh
fi

# Check for docker installation
if ! command -v docker &> /dev/null
then
    echo "ERROR: `docker` not found. Please install docker: https://docs.docker.com/engine/install/"
    exit 1
fi
