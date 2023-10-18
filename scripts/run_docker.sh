#!/bin/sh
# There are several parameters and configurations that need to be run
GPU=(--gpus all)
CONTAINER_NAME="ablator"
ARGS=("${@}")
CMD=()

for ((i=0;i<$#;i++))
do
   case ${ARGS[$i]} in
       # -- option
       --cpu )
       GPU=();;
       --docker-tag )
       ((i++))
       CONTAINER_NAME=${ARGS[$i]};;
       * )
       CMD+=(${ARGS[$i]});;
   esac
done



docker volume create --driver local \
   --opt type=tmpfs \
   --opt device=tmpfs \
   --opt o=uid=0 \
   ${CONTAINER_NAME}-volume
# -v option maps the local docker instance to inside docker
# --cpuset-cpus sets sufficient number of cpus
# --pid option allows access of pids to the host for correct GPU utilization
# --gpus all enables access to GPUs inside docker, remove the option to test without GPUs
docker run --rm -it -v \
   /var/run/docker.sock:/var/run/docker.sock \
   -v ${CONTAINER_NAME}-volume:/ablator \
   -v ${PWD}/shared:/usr/src/app/shared \
   --cpuset-cpus="0-4" \
   --pid host \
   --cap-add SYS_ADMIN \
   --device /dev/fuse \
   --security-opt apparmor:unconfined \
   "${GPU[@]}" ${CONTAINER_NAME} \
   ${CMD[@]}
