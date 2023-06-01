#!/bin/bash

# Define image and container names
IMAGE_NAME="ablator-local-test"
CONTAINER_NAME="ablator-local-test"

# Build the Docker image
docker build -f Dockerfile -t $IMAGE_NAME ..
# Run the Docker container
docker run -it --rm --name $CONTAINER_NAME $IMAGE_NAME
