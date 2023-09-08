GPU=(--gpus all)
CONTAINER_NAME="ablator"
ARGS=("${@}")
DOCKER_RUN=true
BUILD_FLAG=true
for ((i=0;i<$#;i++))
do
   case ${ARGS[$i]} in

       # -- option
       --cpu ) GPU=();;
       --no-docker )
       DOCKER_RUN=false;;
       --no-build )
       BUILD_FLAG=false;;
       --docker-tag )
       ((i++))
       CONTAINER_NAME=${ARGS[$i]};;
       * ) ;;
   esac

done


if [ "$BUILD_FLAG" = true ]; then
   bash scripts/make_docker.sh ${CONTAINER_NAME}
   docker kill $(docker ps -q)
fi

if [ "$DOCKER_RUN" = true ]; then
   mkdir -p shared
   rm -f shared/*
   # -v option maps the local docker instance to inside docker
   # --cpuset-cpus sets sufficient number of cpus
   # --pid option allows access of pids to the host for correct GPU utilization
   # --gpus all enables access to GPUs inside docker, remove the option to test without GPUs
   # --docker-tag is the tagged docker image
   # --runslow runs all pytests
   docker run -v \
      /var/run/docker.sock:/var/run/docker.sock \
      -v ${PWD}/shared:/usr/src/app/shared \
      --cpuset-cpus="0-4" \
      --pid host \
      "${GPU[@]}" ${CONTAINER_NAME} \
      bash ./scripts/run_tests.sh \
      --docker-tag ${CONTAINER_NAME} \
      --no-docker --no-build
else
   # running inside a docker container
   pytest . \
   --docker-tag ${CONTAINER_NAME} \
   --runslow
   if [ ${#GPU[@]} -eq 0 ]; then
      mv coverage.xml shared/coverage_cpu.xml
   else
      mv coverage.xml shared/coverage_gpu.xml
   fi
fi

