# Developer Guide

This guide is meant for those interested in developing on ABLATOR library.

The current document is in progress. The document explains how to set up the development environment and run tests. The repository follows a test-driven development where every PR must include one or more corresponding tests.

**NOTE** Several tests are written for multi-node environments, the minimum requirement for running the full tests is a
machine with 2 GPUs running Ubuntu. The tests can also run with 1 GPU or No GPUs and on Windows and Mac but will not be
comprehensive.

The main library is intended for Prototyping on a local environment e.g. Windows / Mac / Ubuntu but distributed execution
on a multi-node cluster, **only** Ubuntu. When developing features related to a multi-node cluster a  2 < GPU machine will be required.

## Installing the development version of ABLATOR

The development version of Ablator can be installed via pip `pip install -e .[dev]`

The `-e` option automatically updates the library based on the folder contents.

## Setting up Docker environment

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

### For Windows

You will need to install WSL and follow the instructions for Ubuntu (**above** and anywhere in the documentation) even when using Windows. Windows has poor support for
several functionalities that make it non-ideal for the development of distributed applications and thus ablation experiments. The main issues arise when performing the
same tests written for multi-node environments on Windows. This is not an expected use case for the library but if you are developing on Windows you will need a way
to run your tests.

Even when using WSL the biggest issues encountered are the integration of GPU CUDA, Docker and Process Management.
For example, `setproctitle` does not work, `pynvml` has poor support for Windows with several bugs, e.g. [nvmlSystemGetProcessName](https://github.com/gpuopenanalytics/pynvml/issues/49) and Docker has network permission issues between host and container, while `ray` crashes unexpectedly.


**IMPORTANT**
Do not install Windows Docker using `Docker Desktop` for Windows. If you already have please uninstall and follow the instructions above. `Docker `Desktop` is error-prone in the way it communicates with the WSL over the local network and the ray connection breaks randomly and unexplainably.

### For MAC

Homebrew is a package manager for macOS that will make installing Docker and other dependencies easier.
Open a terminal and run the following command to install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Run the following command to install Docker using Homebrew 
```bash 
brew install Docker
```

After installation, you need to start the Docker daemon. Run the following command.
```bash
sudo dockerd &
```
This command will start the Docker daemon in the background.


Docker Compose is a tool for defining and running multi-container Docker applications. You can install it using Homebrew
```bash
brew install docker-compose
```


You can verify that Docker is running by using the following command
```bash
docker --version
```
You should see the version number of Docker if the installation was successful.


To verify whether Docker is running, run the below Docker command in the terminal
```bash
docker run hello-world
```


## Setting up Docker environment for non-root users

You will need to set up docker to run in `root-less` mode. For example, the system user that will be executing the tests should be able to execute: `docker run hello-world` without running into errors. For instructions specific to your system please refer to the [official documentation](https://docs.docker.com/engine/install/linux-postinstall/).

### For Ubuntu
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
You will need to reboot / or log out and log in again for the changes to take effect.

## Building a Docker Image

The tests require the latest ABLATOR image to be built. The image must have the same python version as the running environment.

The easiest way to build a docker image is to run the script **with your **development virtual **environment** active** (as it is used to identify the python version you are using):
`bash script/make_docker.sh`

You will **need** to make the docker image in the main ablator directory **every time** before running the tests (as the code in the docker image is updated from the current repository)

### Details of the Build Process and Docker Instructions (Optional)

You might encounter errors using the script above or you might be working on something that requires you to play around with different python versions. You can inspect [make_docker.sh](scripts/make_docker.sh) or simply play around with:


```bash
docker build --build-arg="PY_VERSION=3.xx.xx" --tag ablator .
```

The [Dockerfile](Dockerfile) is used to build the image.

To run the same image in interactive mode for debugging.

```bash
docker run -it ablator /bin/bash
```

**NOTE** Dockers are ephemeral, any changes you make to the docker container will disappear once the container terminates regardless of what mode you use to execute the container. You can run a container in detached mode by adding the option `-d` which will keep the container active in the background.

```bash
docker run -it -d ablator /bin/bash
```

To connect to an existing image, **first** find the container_id of a running image you want to connect

```bash
docker ps
```

**then**

```bash
docker exec -it <container_id> bash
```


## CUDA Support

**IF** a GPU is detected on the system, Docker tests in ABLATOR will try to start NVIDIA Docker image. To install nvidia container toolkit on your system please refer to the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Optional** (Not Recommended): To disable CUDA for the tests you can set `export CUDA_VISIBLE_DEVICES=''`

To install CUDA:

### For Ubuntu

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo reboot
```
### For Windows
See Above
### For Mac
**TODO**

### Verify installation:

```bash
docker run --rm --runtime=nvidia --gpus all ablator nvidia-smi
```


## Clean Up
During the execution of tests, a mock ray cluster is set up. Due to interruptions or unexpected errors `zombie` docker containers can be left up and running. The zombie containers can interact with tests running on the system and it is **BEST** to terminate all running containers to avoid unwanted interactions.

You can check the currently running docker images with: `docker ps`

To kill / clean all running containers you can run `docker kill $(docker ps -q)`.

It is recommended you do that before every test.

## System dependencies

SSH should be enabled on the current system. It is recommended for security reasons that you configure SSH to be inaccessible outside your local network, [a good guide](https://www.ssh.com/academy/ssh/sshd_config).

The easiest thing would be to disable ssh-server e.g. `sudo systemctl disable ssh` and stop `sudo systemctl stop ssh` when you are not running tests. Additional security options can include preventing access to SSH outside your local network. A risk is when your user account has a weak password, or your ssh-keys are leaked **and** you are connected to an insecure WiFi network.

### For Ubuntu

To disable password login for ssh
You can modify `/etc/ssh/sshd_config` and set

```
PasswordAuthentication no
PubkeyAuthentication yes
```

```bash
sudo apt install openssh-server
sudo systemctl start ssh
sudo systemctl status ssh
```

### For Windows
None
### For Mac

For macOS, enabling and configuring SSH is quite similar to Ubuntu.
You can use the following commands to disable the SSH server and stop the service while running tests on macOS. The below command disables the remote login feature, and effectively disables the SSH server.

```bash
sudo systemsetup -setremotelogin off
```

The below command stops the SSH service using the launchctl command, which manages launch services in macOS.
```bash
   sudo launchctl stop com.openssh.sshd 
```

SSH configuration settings are stored in the /etc/ssh/sshd_config file on macOS, just like on Ubuntu. You can edit the file by using a text editor like nano, vim, or sudo nano.
For example:

```bash
sudo nano /etc/ssh/sshd_config
```

Make the following changes to the sshd_config file - 
- Set PasswordAuthentication to no to disable password-based authentication.
- Set PubkeyAuthentication to yes to enable public key authentication.
After making these changes, save the file (Ctrl + O) and exit (Ctrl + X) the text editor.

Once you've made the changes, you must restart the SSH service to apply them. Use the following commands:
```bash
sudo launchctl stop com.openssh.sshd
sudo launchctl start com.openssh.sshd
```

You can check SSH Status by using the command below:
```bash
sudo launchctl list | grep ssh
```
If the SSH service is running, you'll see an entry indicating its status.

## Testing changes

Make sure that ray is not currently running on your local environment. e.g. by running `ray status`


To test changes you can run them in the main directory:
```bash
pytest .
```

You can also specify additional threads to use for tests e.g.

```bash
pytest -n 10 .
```

where `-n` is the number of parallel tests to run.

As the tests are slow (especially the ones that test for multi-processing) when developing it is a better idea to only run the tests that affect your changes and reserve running all tests at the end of your work. e.g. `pytest -n 10 tests/your_tests.py`

## Contributing Guidelines

To avoid polluting the commit history, each commit should be tested before pushing. Each commit should pass the tests, pylint, mypy and flake8 and have a specific purpose i.e. you should not be making *test commits*, you can experiment in a different branch and then use a separate branch for committing your working changes. This can help other people track the commit history to specific issues in the future.

**NOTE** As there is currently no GPU support in Github actions, you **must** test your code on a machine that has GPUs as well as run your tests inside a Docker container without GPUS. It might seem unnecessary but there have been many cases where test cases fail either when CUDA is present or not present, even if your changes seem unrelated to the entire workflow of the app.

In the main directory (after activating the correct environment):

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
7. black: `black .`

Or simply

`bash scripts/run_test.sh`