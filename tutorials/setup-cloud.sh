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
