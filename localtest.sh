#!/bin/bash

# Define image and container names
IMAGE_NAME="local-test"
CONTAINER_NAME="local-test"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container
docker run -it --rm --name $CONTAINER_NAME $IMAGE_NAME
