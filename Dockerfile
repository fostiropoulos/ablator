FROM continuumio/miniconda3:4.10.3

WORKDIR /usr/src/app

LABEL maintainer="mail@iordanis.me"
LABEL description="Running envrionment for Ablator"

RUN apt-get update && \
    apt-get install -y openssh-server rsync && \
    service ssh start && \
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N "" && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    conda update -y conda && \
    conda install -y python=3.10 pip

COPY ./setup.py ./setup.py
COPY ./README.md ./README.md

RUN pip install -e .[dev]

COPY . .

CMD ["./scripts/starttest.sh"]
