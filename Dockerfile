FROM continuumio/miniconda3:4.10.3

WORKDIR /usr/src/app

LABEL maintainer="deepusc@usc.com"
LABEL version="1.0"
LABEL description="Python application with pytest and Conda"

COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y openssh-server rsync && \
    service ssh start

# Generate SSH keys
RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N "" && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

RUN conda update conda && \
    conda install -y python=3.10 pip

RUN pip install -e .[dev]


# Run pytest when the container launches
CMD ["pytest"]
