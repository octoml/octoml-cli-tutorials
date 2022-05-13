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

We will be installing python dependencies to run this tutorial, so we recommend that you use a Python virtual environment.

```./setup.sh```

  Note the following steps assume that your current working directory is one of the listed example model dirs above. Let's `cd` into the `vision` repository for example. Also make sure you've globally installed the OctoML CLI, by moving the CLI binary to an appropriate location like /usr/local/bin or adding it to your PATH.

## Local inference without a container

Before deploying this model to a production-scale microservice, let's make sure we can run a single inference locally on our model:

```python run.py --local```

The `critterblock.onnx` model is a Computer Vision (Resnet-based) model customized for a cat door; the door stays closed when it detects that a cat is carrying its prey and opens when the cat is not. In `run.py`, we pass in a sample image to the model, run image preprocessing code customized for this use case, run an inference using ONNX Runtime, and finally call image post-processing code on the results of the model. The script returns a cat score of 1, approach score of 1, and prey score of 0.999 on this image, which means the model correctly detected that a cat is approaching the cat door while holding its prey.


## Generate production-scale deployment using OctoML CLI
Now that we've confirmed the model is working as intended, let's prepare it for production-scale usage. We will deploy the package locally using the OctoML CLI without having to upload our models to the OctoML platform.

Generate docker package + start triton container -- this will deploy a docker container.

```octoml deploy```

After running the command above, you can verify that a Docker container has been spun up successfully by running `docker ps`.

## Confirm that we can run inferences on the container locally.

By default, the running Docker container exposes a GRPC endpoint at port 8001. The
following invocation will run the client code for sending a sample inference request
to that default port. `run.py` contains client code for the deployed docker container.

```python run.py --triton```

Again, we see a cat score of 1, approach score of 1, and prey score of 0.999 on our sample image. This means our containerized model is working as intended.

## Now that we have a production-grade container ready for deployment, let's deploy the container to an existing EKS cluster.
EKS is AWS's Kubernetes service for production scale services

Note: the following steps assume the following have already been configured:

- An AWS Elastic Container Registry (ECR) for pushing built docker images to
- An EKS cluster with a node pool for c5n.xlarge instances that has a no_schedule taint for `octoml.ai/octomizer-platform: aws-c5n.xlarge`
- nginx-ingress for above EKS cluster
- IAM access to the EKS cluster

Navigate one level back up to the tutorials repo. Then, run the following script to push our local container to an AWS ECR repo and install 
a sample helm chart to the EKS cluster.
```./deploy_to_eks.sh <model_name> <aws_profile_name> <docker_image_tag_you_want_to_push> <aws_registry_url> <aws_cluster_name> <aws_region>```

For example:

```./deploy_to_eks.sh critterblock 186900524924_Sandbox-Developer critterblockv5 186900524924.dkr.ecr.us-west-2.amazonaws.com octoml-sandbox-sxq590bv us-west-2```

## Finally, run inferences on the container deployed to EKS

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
