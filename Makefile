#Makefile for the Ablator
.PHONY: default unittest pylint mypy all build down

default: all

unittest: build
	-docker-compose run unittest

pylint: build
	-docker-compose run pylint

mypy: build
	-docker-compose run mypy

build: build
	DOCKER_BUILDKIT=1 docker-compose build builder

down:
	docker-compose down

all: build unittest pylint mypy down
