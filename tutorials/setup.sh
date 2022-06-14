#!/bin/bash
set -eux

# Install pip requirements
pip install -r requirements.txt

# Check for docker installation
if ! command -v docker &> /dev/null
then
    echo "ERROR: docker not found. Please install docker: https://docs.docker.com/engine/install/"
    exit 1
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

if [[ -n "$MODEL" ]]; then
  ./download_model.sh "$MODEL"
fi
