# Tutorials

The tutorials in this directory will walk you through how to:

- Create an ML model container using the `octoml` CLI
- Local container deployment and inference via docker
- Remote container deployment and inference via Amazon Elastic Kubernetes Service (EKS)
- Accelerating a model to save on cloud costs by connecting to the OctoML platform

Three example model setups are provided for you to play with:

```
- vision/                 -- A classification vision model in ONNX to detect whether a cat is trying to bring their prey into your house
- question_answering/    -- BERT in ONNX, using the huggingface transformers library for pre and post processing
- generation/             -- GPT2 in ONNX, calling into the huggingface transformers library for generation at runtime
```

## Setup Environment

In this section we will validate the environment and install some dependencies. Make sure you've globally installed the OctoML CLI, by moving the CLI binary to an appropriate location like `/usr/local/bin` or adding it to your `$PATH`.

Verify you have the `octoml` cli installed:

```shell
$ octoml -V
octoml 0.2.4 (d5d534e 2022-05-14 23:38:45)
```

Verify you have python installed. You need Python3. Python3.10 users (e.g. Ubuntu Jammy) see our [Note on Python3.10](https://github.com/octoml/octoml-cli-tutorials/tree/main/tutorials/vision#note-on-python310)

```shell
$ python --version
Python 3.8.4
```

Verify you have a running docker daemon:

```shell
docker ps
```

Clone the tutorials repo:

```shell
git clone https://github.com/octoml/octoml-cli-tutorials
cd octoml-cli-tutorials
cd tutorials
```

We recommend that you use a Python virtual environment:

```shell
python -m venv venv
source venv/bin/activate
```

Run setup utility to install the python dependencies.

```shell
./setup.sh
```

> Note: The following steps assume that your current working directory is one of the listed example model dirs above. Let's `cd` into the `vision` repository for example. 

Change Directory into the vision tutorial:

```
$ cd vision
$ ls
cat_input_images/  critterblock.onnx  octoml.yaml  run.py
```

## Local inference without a container

Before deploying this model to a production-scale microservice, let's make sure we can run a single inference locally on our model:

```
$ python3 run.py --local

Cat score: 1.000
Approach score: 1.000
Prey score: 0.999
--------------------------------------
```

The `critterblock.onnx` model is a Computer Vision (Resnet50-based) model customized for a cat door; the door locks when it detects that a cat is carrying its prey into the house. In `run.py`, we pass in a sample image to the model, run image preprocessing code customized for this use case, run an inference using ONNX Runtime, and finally call image post-processing code on the results of the model. In this case, the script returns a cat score of 1, approach score of 1, and prey score of 0.999 on this image, which means the model correctly detected that a cat is approaching the cat door while holding its prey.


## Generate production-scale deployment using OctoML CLI

Now that we've confirmed the model is working as intended, let's prepare it for production-scale usage (in this case, so that we can protect thousands of cloud-connected cat doors). We will deploy the package locally using the OctoML CLI without having to upload our models to the OctoML platform.

The `octoml.yaml` file has been created for you already:

```
$ cat octoml.yaml
---
models:
  critterblock:
    path: critterblock.onnx
```

Now use this file to generate a docker image and start a docker container running triton -- this will deploy a docker container locally on your machine.

```
$ octoml deploy
 ∙∙∙ Models imported
 ∙∙∙ Packages generated
 ∙∙∙ Docker image assembled
 ∙∙∙ Triton image built
 ∙∙∙ 🐳 Triton container started

A docker container with a triton inference server is now running on your local
machine. For more information about how to interact with your model, please
refer to our tutorials and guides on GitHub.

https://github.com/octoml/octoml-cli-tutorials
```

After running the command above, you can verify that a Docker container has been spun up successfully by running `docker ps`.

```
$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED      STATUS      PORTS                              NAMES
7182ac8ee475   a85ac5657dc0   "/opt/tritonserver/n…"   3 days ago   Up 3 days   0.0.0.0:8000-8002->8000-8002/tcp   vengeful-coach
```

## Confirm that we can run inferences on the container locally.

By default, the running Docker container exposes a GRPC endpoint at port 8001. The
following invocation will run the client code for sending a sample inference request
to that default port. `run.py` contains client code for the deployed docker container.

```
$ python3 run.py --triton

Cat score: 1.000
Approach score: 1.000
Prey score: 0.999
--------------------------------------
```

Again, we see a cat score of 1, approach score of 1, and prey score of 0.999 on our sample image. This means our containerized model is working as intended.

## Kubernetes Deployment

Now that we have a production-grade container ready for deployment, let's deploy the container to an existing EKS cluster. EKS is AWS's Kubernetes service for production scale services.

The steps below assume that the following have already been configured:

- An EKS cluster with a node pool for c5n.xlarge instances
- IAM access to the EKS cluster

Navigate one level back up to the tutorials repo. 

```shell
$ cd ..
$ ls
README.md  deploy_to_eks.sh  generation  helm_chart  question_answering  requirements.txt  setup-cloud.sh  setup.sh  vision
```

The following script requires `kubectl`, `helm`, and `aws` to be installed. You can run `./setup-cloud.sh` to install them. This requires `sudo`.

Install cloud utilities (if needed):

```
./setup-cloud.sh
```

Fill in these bash variables for convenience:

```
model_name=
aws_profile_name=
docker_image_tag=
aws_registry_url=
cluster_name=
aws_region=
```

For example:

```
model_name=critterblock
aws_profile_name=186900524924_Sandbox-Developer
docker_image_tag=v5
aws_registry_url=186900524924.dkr.ecr.us-west-2.amazonaws.com
cluster_name=octoml-sandbox-sxq590bv
aws_region=us-west-2
```

Run the following script to push the local container to an AWS ECR repo and install a sample helm chart to the EKS cluster:


```
./deploy_to_eks.sh ${model_name} ${aws_profile_name} ${docker_image_tag} ${aws_registry_url} ${cluster_name} ${aws_region}
```

Check the status of the helm deployment:

```
helm list -n ${model_name}
```

## Inference on Cloud

Finally, run inferences on the container deployed to EKS

Once the helm deploy completes, port-forward the EKS deployment's triton GRPC endpoint to localhost:

```
kubectl port-forward service/${model_name} -n ${model_name} 8080:8001
```

Now we can run inference:

```
python run.py --triton --port 8080
```

To run inference with the http endpoint, port-forward the HTTP endpoint instead:

```
kubectl port-forward service/${model_name} -n ${model_name} 8080:8000
```

And run inference with http:

```
python run.py --triton --protocol http --port 8080
```


To clean up the environment or if you want to try other live demos on EKS, stop the port-forward script and run:

```
helm uninstall ${model_name} -n ${model_name}
```

## Accelerating your model on different hardware targets

To access advanced features like model acceleration, you will need to [sign up](https://learn.octoml.ai/private-preview) for an OctoML Platform account. Once you've submitted the signup form, you will receive an email within 1 business day with instructions on how to finish setting up your account. Next, generate an API access token [here](https://app.octoml.ai/account/settings) and call `octoml setup acceleration` to store your API access token in the CLI.

`octoml setup acceleration` is an interactive help wizard that not only prompts you for the API access token but also helps you populate the input configuration file (`octoml.yaml`) with other fields required for acceleration, including hardware and (for dynamically shaped models only) the model's input shapes. Beware that hardware selection requires that you click <space> and then <return> to confirm your selection.


```shell
$ octoml setup acceleration

Updated octoml.yaml with the new hardware targets aws_c5.12xlarge - Cascade Lake - 48 vCP
```
  

Now, you are ready to run accelerated packaging, which returns you the best-performing package with minimal latency for each hardware you've selected, after exploring multiple acceleration strategies including TVM, ONNX-RT, and the native training framework. We recommend running express mode acceleration as follows, which completes within 20 minutes. If you are willing to wait for several hours for potentially better latency, run `octoml package -a` for full acceleration mode.


```shell
$ octoml package -e
```

Now, verify that you've successfully built an accelerated container. 

```shell
$ docker images | head
```

You can now push the local container to a remote container repository (e.g. ECR) per the instructions above, then run inferences against the container on a remote machine with an architecture matching the one you've accelerated the model for.

If you wish to locally deploy and test inferences against the accelerated container, you may run the following command, but note that it only works if the local machine on which you're running the CLI has the same hardware architecture as the hardware you accelerated the model for (e.g. if you are running the CLI on an M1 mac, you can only run deployment on your local mac successfully if you accelerated your model on a Graviton instance, as both of them share the ARM64 architecture).

```shell
$ octoml deploy -e
```

## Troubleshooting

To check for Kubernetes deployment info, run:

```
kubectl get all -n ${model_name}
```

To get the logs for a failed pod deployment, run the above and modify the pod name in the following command:

```
kubectl logs pod/demo-6f45998bbb-6jnlq -n ${model_name}
```

For Torchscript models traced on a GPU, containers will not be able to be run on CPUs in the local CLI. Please sign up for an OctoML account and upgrade to authenticated usage per the instructions above if this use case is applicable for you.
