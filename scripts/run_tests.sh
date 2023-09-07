GPU=(--gpus all)
CONTAINER_NAME="ablator"
for i in $@
do

   case $i in

       # -- option
       --cpu ) GPU=();;
       * ) CONTAINER_NAME=$i;;
   esac

done
bash scripts/make_docker.sh ${CONTAINER_NAME}
docker kill $(docker ps -q)
# -v option maps the local docker instance to inside docker
# --cpuset-cpus sets sufficient number of cpus
# --pid option allows access of pids to the host for correct GPU utilization
# --gpus all enables access to GPUs inside docker, remove the option to test without GPUs
# --docker-tag is the tagged docker image
# --runslow runs all pytests
docker run -v \
   /var/run/docker.sock:/var/run/docker.sock \
   --cpuset-cpus="0-4" \
   --pid host \
   "${GPU[@]}" ${CONTAINER_NAME} \
   pytest . \
   --docker-tag ${CONTAINER_NAME} \
   --runslow
