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

COPY . .
RUN --mount=type=cache,target=/root/.cache \
    pip install -e .[dev]

EXPOSE 22
RUN chmod a+x ./scripts/docker-entrypoint.sh
ENTRYPOINT ["./scripts/docker-entrypoint.sh"]

CMD ["pytest","."]
