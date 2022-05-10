# Demos

The demos in this directory will walk you through:

- ML model container generation using the `octoml` CLI
- Container deployment and inference locally via docker
- Container deployment and inference via Amazon Elastic Kubernetes Service (EKS)

Three example model setups are provided for you to play with:

```
- vision/                 -- A classification vision model in ONNX to detect whether a cat is bringing in prey
- question\_answering/    -- BERT in ONNX, using the huggingface transformers library for pre and post processing
- generation/             -- GPT2 in ONNX, calling into the huggingface transformers library for generation at runtime
```

## Install requirements

```./setup.sh```

  Note the following steps assume your current working directory is one of the listed example model dirs above.

## Run inference locally

```python run.py --local```

## Generate local deployment without authenticating to the OctoML Platform

Generate docker package + start triton container -- this will deploy a docker container.

```octoml deploy```

## Run inference via local triton container -- `run.py` contains client code for the
deployed docker container.

By default, the running Docker container exposes a GRPC endpoint at port 8001. The
following invocation will run the client code for sending a sample inference request
to that default port.

```python run.py --triton```

## Deploy a triton docker build to an existing EKS cluster.

Note: the following steps assume the following have already been configured:

- An AWS Elastic Container Registry (ECR) for pushing built docker images to
- An EKS cluster with a node pool for c5n.xlarge instances that has a no_schedule taint for `octoml.ai/octomizer-platform: aws-c5n.xlarge`
- nginx-ingress for above EKS cluster
- IAM access to the EKS cluster

For sandbox access key, go to Okta AWS SSO -> octoml-sandbox -> get credentials (option 2) -> paste to ~/.aws/credentials + Take note of the profile name that was used.

## Run script to deploy to EKS
```./deploy_to_eks.sh <model_name> <aws_profile_name> <docker_image_tag_you_want_to_push> <aws_registry_url> <aws_cluster_name> <aws_region>```

For example:

```./deploy_to_eks.sh critterblock 186900524924_Sandbox-Developer critterblockv5 186900524924.dkr.ecr.us-west-2.amazonaws.com octoml-sandbox-sxq590bv us-west-2```

## Run inference via triton container deployed to eks

The last step of the `deploy_to_eks.sh` script port-forwards the EKS deployment's triton GRPC endpoint to localhost

```kubectl port-forward service/<model_name> -n <model_name> 8080:80```

Now we can run inference:

```python run.py --triton --port 8080```

To clean up the environment or if you want to try other live demos on EKS, kill the port-forward script and run:

```helm uninstall demo```

## Troubleshooting

To check for Kubernetes deployment info, run

```kubectl get all -n demo```

To get the logs for a failed pod deployment, run the above and modify the pod name below in the following command:

```kubectl logs pod/demo-6f45998bbb-6jnlq -n demo```
