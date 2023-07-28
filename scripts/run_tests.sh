bash scripts/make_docker.sh ${1:-ablator}
docker run -v \
   /var/run/docker.sock:/var/run/docker.sock \
   --cpuset-cpus="0-4" \
   --pid host \
   --gpus all ${1:-ablator} \
   pytest . \
   --docker-tag ${1:-ablator} \
   --runslow
