# FROM nvidia/cuda:11.5.2-base-ubuntu20.04
FROM continuumio/miniconda3:4.10.3
WORKDIR /usr/src/app

LABEL maintainer="mail@iordanis.me"
LABEL description="Running environment for Ablator"

RUN apt-get update
RUN apt-get install -y openssh-server rsync
RUN service ssh start
RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN conda update -y conda
ARG PY_VERSION=3.10.12
RUN conda install -y python=$PY_VERSION pip

COPY . .

RUN pip install -e .[dev]


EXPOSE 22
CMD ["pytest","."]
