#!/bin/bash

length=5
random_string=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w "$length" | head -n 1)

IMAGE_NAME="ablator-local-test-${random_string}"
CONTAINER_NAME="ablator-local-test-${random_string}"

# Build the Docker image
docker build -f Dockerfile -t $IMAGE_NAME ..
# Run the Docker container
docker run -it --rm --name $CONTAINER_NAME $IMAGE_NAME
# Delete the Docker image after the container exits
docker rmi "$IMAGE_NAME"