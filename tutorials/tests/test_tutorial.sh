#!/bin/bash
set -ex
TUTORIAL_DIR=$1

# Local test
cd "$TUTORIAL_DIR"
python run.py --local
# Triton native GRPC test
python run.py --triton
# Triton python GRPC test
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run.py --triton
# Triton http test
python run.py --triton --protocol http
