# OctoML CLI

The `octoml` CLI helps you create deployable containers for your ML models using the OctoML Platform, in the terminal or within your CI/CD pipeline.
You can [deploy and run](https://github.com/octoml/octoml-cli-tutorials/tree/main/tutorials#demos) inferences on the container locally for development and testing, then deploy the same container to the cloud.

By use of this CLI application, you agree to OctoML’s [Terms of Service](https://octoml.ai/legals/terms-of-service/) and [Privacy Policy](https://octoml.ai/legals/privacy-policy/).

![OctoML CLI Example](https://www.datocms-assets.com/45680/1652749860-octoml_cli.gif)

This repository contains [multiple examples](https://github.com/octoml/octoml-cli-tutorials/tree/main/tutorials#demos) of deploying OctoML containers to any cloud environment given an image registry,
and K8s cluster, but the containers can be used in any environment that supports Docker.

**Model framework coverage**: We support TensorFlow SavedModel, TensorFlow GraphDef, Torchcript (PyTorch), and ONNX models.

**Hardware coverage**: You may run our CLI on x86, CUDA machines, and ARM64 machines (including M1 Macs). In unauthenticated usage, the containers built via the CLI may only be deployed on hardware architectures matching the local machine on which you ran `octoml package`. In authenticated usage, you may create accelerated model containers deployable to any x86, CUDA, and ARM64 remote cloud instance, even if that hardware architecture does not match that of your local machine on which you ran `octoml package`.

**OS coverage**: We support MacOS, Linux (Ubuntu 18.04+), and Windows. If you wish to use the CLI on Ubuntu 18.04, please ensure your protobuf version is on 3.19.4.

## Getting started

To get started all you need to do is download the latest version of the OctoML CLI using `wget` or `curl`.

Once you have the tool you should be able to view the list of available commands by running `./octoml` in your current directory.

To globally install move to an appropriate location like `/usr/local/bin` or add to your `PATH`.

You can now jump straight into our [tutorials](https://github.com/octoml/octoml-cli-tutorials/tree/main/tutorials#demos)

## Core commands

**Note: that you do not need to provide an OctoML API access token at this stage.**

`octoml init`: This is the first command we recommend that you run. It helps you set up an input configuration file by prompting you for the information required for the CLI to generate a container for deployment.

`octoml package`: Builds a deployment-ready Docker container for the models specified in the input configuration file. The input configuration file is required for this command to complete successfully.

`octoml deploy`: Deploys a Docker container to a locally hosted endpoint. After completion of this command, you may run `docker ps` to confirm the a container has been successfully generated for you.

**You can run the commands above with or without an OctoML account. If you run them without setting an OctoML API token beforehand, we do not upload your model to the OctoML platform.**

## Sign up for an OctoML account/ authenticate to access advanced features including model acceleration and benchmarking
OctoML combines state-of-the-art compiler technologies (TVM, ONNX-RT, and others) to give you the best-performing package for any model. To access OctoML's acceleration and benchmarking services, you will need to [sign up for an OctoML account](https://learn.octoml.ai/private-preview) and create an API token using the OctoML web UI.

`octoml setup acceleration`: Prompts you for information required for acceleration, including an OctoML API access token, hardware, and dynamic shape disambiguation. Populates the information into your input configuration file for downstream use in `octoml package` and `octoml deploy`. If you do not wish to do so, please make sure to configure your input configuration file manually with the requisite fields for acceleration before calling `octoml package -e` or `octoml package -a`.

`octoml package -e`: We recommend running acceleration in express mode, which completes within 20 minutes. If you are willing to wait for several hours for potentially better latency via a fuller exploration of the optimization space, run `octoml package -a` for full acceleration mode. Both modes return the best-performing package with minimal latency for each hardware you've selected, after exploring multiple acceleration strategies including TVM, ONNX-RT, and the native training framework. After this step we recommend that you directly send this built container to a remote cloud repository for downstream remote usage.

`octoml deploy -a` or `octoml deploy -e`: Only run this command if you're on hardware with an architecture matching the cloud instance for which you have accelerated the model. If you previously ran `package` on more than one hardware target, make sure to call `deploy` with a specific hardware input.

## Deploying OctoML packages to production-grade microservices and applications

See the [tutorials](https://github.com/octoml/octoml-cli-tutorials/tree/main/tutorials#demos) folder for how to deploy OctoML model containers to downstream cloud registries like ECR and Kubernetes services like EKS.

## Telemetry

OctoML by default collects telemetry on your usage of the CLI. However, we do not collect your specific model unless you decide to sign up for access to the OctoML platform and submit your OctoML API access token.

The first time you run the CLI's `octoml init` command, you will get prompted on whether you'd like to opt out of telemetry. You can also decide to opt out of telemetry anytime by setting the following environment variable: `export OCTOML_TELEMETRY=false`.
