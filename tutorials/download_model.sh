#!/bin/bash
set -eux

PYTHON=python
# Compat for python3
if ! command -v python &> /dev/null
then
    PYTHON=python3
fi

MODEL="all"
for i in "$@"; do
  case $i in
    -m=*|--model=*)
      MODEL="${i#*=}"
      shift # past argument=value
      ;;
    --*|-*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

if [[ "$MODEL" == "all" || "$MODEL" == "qa" ]]; then
# Get the distilbert model for the question answering demo
  [ -f question_answering/model.onnx ] || ${PYTHON} -m transformers.onnx --model=distilbert-base-uncased-distilled-squad --feature=question-answering question_answering
fi

if [[ "$MODEL" == "all" || "$MODEL" == "gpt" ]]; then
# Get the gpt2 model for the generation model
  [ -f generation/model.onnx ] || ${PYTHON} -m transformers.onnx --model=distilgpt2 --feature=causal-lm generation
fi
