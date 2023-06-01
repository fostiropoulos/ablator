FROM continuumio/miniconda3:4.10.3

WORKDIR /usr/src/app

LABEL maintainer="deepusc@usc.com"
LABEL description="Local test for Ablator"

COPY . .

RUN apt-get update && \
    apt-get install -y openssh-server rsync && \
    service ssh start

RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N "" && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

RUN conda update conda && \
    conda install -y python=3.10 pip

RUN pip install -e .[dev]

RUN conda install -y pytest

RUN chmod +x ./hack/starttest.sh

# CMD ["sh", "-c","/usr/sbin/sshd -D & && pytest ."]
CMD ["./hack/starttest.sh"]
