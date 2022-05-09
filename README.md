# OctoML CLI
The `octoml` CLI helps you generate deployable containers for your ML models using the OctoML Platform, in the terminal or within your CI/CD pipeline. We also provide a convenience function for you to deploy the container for local testing/inferencing. If you are happy with the local results, you can push the container we've created to a remote cloud registry and K8 service of your choice, following our example scripts.

We support TensorFlow SavedModel, TensorFlow GraphDef, Torchcript (PyTorch), and ONNX models.

## Getting started
Windows: `wget downloads.octoml.ai/octoml.zip`\
MacOS: `wget downloads.octoml.ai/octoml_macOS.tar.gz`\
Linux: `wget downloads.octoml.ai/octoml_ubuntu.tar.gz`

## Core commands for unauthenticated usage
Once you have downloaded the `octoml` CLI binary and dragged it to your desired working directory, you can view a list of available commands by running `./octoml` in the directory:

`octoml init`: This is the first command we recommend that you run. It helps you set up an input configuration file by prompting you for the information required for the CLI to generate a container for deployment. Note that you do not need to provide an OctoML API access token at this stage.

`octoml package`: Generates a deployment-ready Docker container for the models specified in the input configuration file. The input configuration file is required for this command to complete successfully. 

`octoml deploy`: Deploys a Docker container to a locally hosted endpoint. After completion of this command, you may run `docker ps` to confirm the a container has been successfully generated for you.

## Sign up for an OctoML account/ authenticate to access advanced features including model acceleration and benchmarking
OctoML combines state-of-the-art compiler technologies (TVM, ONNX-RT, and others) to give you the best-performing package for any model. To access OctoML's acceleration and benchmarking services, you will need to sign up for an OctoML account and create an API token using the OctoML web UI.

`octoml setup acceleration`: Prompts you for information required for acceleration, including an OctoML API access token, hardware, dynamic shape disambiguation, and a choice of express mode (completes within 20 minutes) versus full acceleration (may take several hours). 

`octoml package`: After running `octoml setup acceleration`, your input configuration file will have additional fields required for acceleration. The package command will parse those fields, attempt multiple acceleration strategies, then generate a deployable container for your model with minimal latency.

`octoml deploy`: Same as above. You may now use this command to deploy your best-performing model container to a locally hosted endpoint.

## Deploying OctoML packages to production-grade microservices and applications

See the examples folder for how to deploy OctoML model containers to downstream cloud registries like ECR and Kubernetes services like EKS.
