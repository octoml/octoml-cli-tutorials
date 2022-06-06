# OctoML CLI

The `octoml` CLI helps you create deployable containers for your ML models using the OctoML Platform, in the terminal or within your CI/CD pipeline.
You can deploy and run inferences on the container locally for development and testing, then deploy the same container to the cloud.

By use of this CLI application, you agree to OctoML’s [Terms of Service](https://octoml.ai/legals/terms-of-service/) and [Privacy Policy](https://octoml.ai/legals/privacy-policy/).

![OctoML CLI Example](https://www.datocms-assets.com/45680/1652749860-octoml_cli.gif)

This repository contains multiple examples of deploying OctoML containers to any cloud environment given an image registry,
and K8s cluster, but the containers can be used in any environment that supports Docker.

**Model framework coverage**: We support TensorFlow SavedModel, TensorFlow GraphDef, Torchcript (PyTorch), and ONNX models.

**Hardware coverage**: We support x86, CUDA machines, and ARM64 machines (including M1 Macs).

**OS coverage**: We support MacOS, Linux (Ubuntu 18.04+), and Windows. If you wish to use the CLI on Ubuntu 18.04, please ensure your protobuf version is on 3.19.4.

## Getting started

To get started all you need to do is download the latest version of the OctoML CLI using `wget` or `curl`.

Once you have the tool you should be able to view the list of available commands by running `./octoml` in your current directory.

To globally install move to an appropriate location like `/usr/local/bin` or add to your `PATH`.

## Core commands

**Note: that you do not need to provide an OctoML API access token at this stage.**

`octoml init`: This is the first command we recommend that you run. It helps you set up an input configuration file by prompting you for the information required for the CLI to generate a container for deployment.

`octoml package`: Generates a deployment-ready Docker container for the models specified in the input configuration file. The input configuration file is required for this command to complete successfully.

`octoml deploy`: Deploys a Docker container to a locally hosted endpoint. After completion of this command, you may run `docker ps` to confirm the a container has been successfully generated for you.

**You can run the commands above with or without an OctoML account. If you run them without setting an OctoML API token beforehand, we do not upload your model to the OctoML platform.**

## Sign up for an OctoML account/ authenticate to access advanced features including model acceleration and benchmarking
OctoML combines state-of-the-art compiler technologies (TVM, ONNX-RT, and others) to give you the best-performing package for any model. To access OctoML's acceleration and benchmarking services, you will need to sign up for an OctoML account and create an API token using the OctoML web UI.

`octoml setup acceleration`: (Coming soon) Prompts you for information required for acceleration, including an OctoML API access token, hardware, dynamic shape disambiguation, and a choice of express mode (completes within 20 minutes) versus full acceleration (may take several hours).

`octoml package`: After running `octoml setup acceleration` (coming soon), your input configuration file will have additional fields required for acceleration. The package command will parse those fields, attempt multiple acceleration strategies, then generate a deployable container for your model with minimal latency. If you wish to run `octoml package` in accelerated mode without first calling `octoml setup acceleration`, please make sure to configure your input configuration file with the required fields.

`octoml deploy`: Same as above. You may now use this command to deploy your best-performing model container to a locally hosted endpoint.

## Deploying OctoML packages to production-grade microservices and applications

See the tutorials folder for how to deploy OctoML model containers to downstream cloud registries like ECR and Kubernetes services like EKS.

## Telemetry

OctoML by default collects telemetry on your usage of the CLI. However, we do not collect your specific model unless you decide to sign up for access to the OctoML platform and submit your OctoML API access token.

The first time you run the CLI's `octoml init` command, you will get prompted on whether you'd like to opt out of telemetry. You can also decide to opt out of telemetry anytime by setting the following environment variable: `export OCTOML_TELEMETRY=false`.
ohai
