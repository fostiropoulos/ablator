bash scripts/make_docker.sh
docker run -v \
   /var/run/docker.sock:/var/run/docker.sock \
   --cpuset-cpus="0-4" \
   --pid host \
   --gpus all \
   ablator
