docker_tag="ablator"
docker_build_args=""
test_tag=""
tests="."


.PHONY: test
test-fast:
	# running inside a docker container
	pytest ${tests} \
	--docker-tag ${docker_tag} \
	--reruns 2 \
	--reruns-delay 10 \
	--fast \
	-n=0

test-slow:
	# running inside a docker container
	pytest ${tests} \
	--docker-tag ${docker_tag} \
	--reruns 2 \
	--reruns-delay 10 \
	--slow \
	-n=0 ${test_tag}

in-docker-test:
	# for running tests inside a docker container we must
	# specify volume and mv coverage file on shared directory
	pytest ${tests} \
	--docker-tag ${docker_tag} \
	--volume-name ${docker_tag}-volume \
	--reruns 2 \
	--reruns-delay 10 \
	-n=0 ${test_tag}
	mv coverage.xml shared/_coverage.xml

clean-docker:
	-docker kill $(docker ps --filter ancestor=${docker_tag} -q)
	docker container prune -f
	-docker volume rm ${docker_tag}-volume

docker:
	mkdir -p shared
	bash scripts/make_docker.sh ${docker_build_args} ${docker_tag}

run-docker: docker
	bash scripts/run_docker.sh --it --docker-tag ${docker_tag} bash

run-docker-clean: clean-docker docker
	bash scripts/run_docker.sh --it --docker-tag ${docker_tag} bash

docker-test: clean-docker docker
	bash scripts/run_docker.sh --docker-tag ${docker_tag} \
	make in-docker-test test_tag="${test_tag}" docker_tag="${docker_tag}" tests="${tests}"
	mv shared/_coverage.xml shared/coverage_gpu.xml

docker-test-cpu: clean-docker docker
	bash scripts/run_docker.sh --cpu --docker-tag ${docker_tag} \
	make in-docker-test test_tag="${test_tag}" docker_tag="${docker_tag}" tests="${tests}"
	mv shared/_coverage.xml shared/coverage_cpu.xml

install:
	pip install -e ."[dev]" -v

flake8:
	flake8 ./ablator/ --count --show-source --statistics
	flake8 --ignore=F841,W503,DOC ./tests/

black:
	black --check --preview .

pylint:
	pylint ablator

mypy:
	mypy ablator

static-tests: black flake8 mypy pylint
	echo "Done"

package:
	bash scripts/package.sh
