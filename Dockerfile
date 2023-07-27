FROM continuumio/miniconda3:4.10.3
WORKDIR /usr/src/app

LABEL maintainer="mail@iordanis.me"
LABEL description="Running environment for Ablator"

RUN apt-get update
RUN apt-get install -y openssh-server rsync
RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -N ""
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN conda update -y conda
ARG PY_VERSION=3.10.12
RUN conda install -y python=$PY_VERSION pip

# This is done to avoid re-installing depedencies on code changes.
COPY ./setup.py ./setup.py
COPY ./README.md ./README.md
RUN pip install -e .[dev]
COPY . .

EXPOSE 22
RUN chmod a+x ./scripts/docker-entrypoint.sh
ENTRYPOINT ["./scripts/docker-entrypoint.sh"]

CMD ["pytest","."]
