# Developer Guide

This guide is meant for those interested in developing on ABLATOR library.

The current document is in progress.
## Installing Development Version of Ablator

The development version of Ablator can be installed via pip `pip install -e .[dev]`

The `-e` option automatically updates the libraries content

## Setting up Docker Enviroment

Docker is used for running tests and is required to be installed. For detailed instructions on how to install Docker please refer to the [official documentation](https://docs.docker.com/engine/install/).

### For Ubuntu
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```
## Setting up Docker Enviroment for non-root users

You will need to set-up docker to run in `root-less` mode. For example, the system user that will be executing the tests should be able to execute: `docker run hello-world` without running into errors. For instructions specific to your system please refer to the [official documentation](https://docs.docker.com/engine/install/linux-postinstall/).

### For Ubuntu
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
You will need to reboot / or log-out and log-in again for the changes to take effect.

## Building a Docker Image

The tests require the latest ABLATOR image to be build. The image must have the same pythong version as the running enviroment. From the Dockerfile you will need to modify `python=xxx` to the version of python currently present in your test enviroment.
e.g.

`python --version`
> Python 3.xx.xx

To build the docker image you will need to execute in the main ablator directory before running the tests.

```bash
docker build --build-arg="PY_VERSION=3.xx.xx" --tag ablator .
```


or automatically (**NOTE** your test enviroment would need to be active to correctly identify the python version)

```bash
docker build --build-arg="PY_VERSION=$(python --version | grep -Eo '[0-9]\.[0-9]+\.[0-9]+')" --tag ablator .
```

You can run the same image in interactive mode:

```bash
docker run -it ablator /bin/bash
```

## CUDA Support

IF a GPU is detected on the system, Docker tests will try to start NVIDIA Docker image. To install nvidia container toolkit on your system please refer to the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Optional** (Not Recommended): to disable this behavior you can set `export CUDA_VISIBLE_DEVICES=''`

### For Ubuntu

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit-base

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo reboot
```
#### Verify installation:
```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Clean Up

During execution of tests, a mock ray cluster is set-up. Due to interuptions or unexpected errors `zombie` nodes can be left up and running.

You can check the currently running docker images with: `docker ps`

To kill / clean all running containers you can run `docker kill $(docker ps -q)`.

It is recommended you do that before every test.


## Testing changes

To test changes you can run in the main directory:
```bash
pytest .
```

You can also specify additional threads to use for tests e.g.

```bash
pytest -n 10 .
```

where `-n` is the number of parallel tests to run.

As the tests are slow (especially the ones that test for multi-processing) when developing it is a better idea to only run the tests that affect your changes and reserve running all tests at the end of your work. e.g. `pytest -n 10 tests/your_tests.py`

## Contributing Guide-Lines

To avoid polluting the commit-history, each commit should be tested prior to pushing. Each commit should pass the tests, pylint, mypy and flake8.

**NOTE** as there is currently no GPU support in Github actions, you **must** test your code on a machine that has GPUs as well as running inside a Docker container without GPUS. It might seem uneccessary but there have been many cases were test cases fail for either when CUDA is present or not present, even if your changes seem unrelated to the entire workflow of the app.

 In the main directory (after activating the correct enviroment):
1. `bash scripts/make_docker.sh`
2.
```bash
# maps the local docker instance to inside docker
# sets sufficient number of cpus
# allows access of pids to the host for correct GPU utilization
# enables access to GPUs inside docker, remove `--gpus all` to test without GPUs
# `ablator` is the tagged docker
docker run -v \
   /var/run/docker.sock:/var/run/docker.sock \
   --cpuset-cpus="0-4" \
   --pid host \
   --gpus all \
   ablator
```
3. pylint: `pylint ablator`
4. mypy: `mypy ablator`
5. flake8: `flake8 ablator`
6. pydoc-lint: `pydoclint ablator`

Or simply

`bash scripts/run_test.sh`