#Makefile for the Ablator
.PHONY: default
default: unit-test

.PHONY: unit-test
unit-test:
	LENGTH=5;\
	DOCKERFILE=./localtest/Dockerfile;\
	BUILD_CONTEXT=.;\
	RANDOM_STRING=$$(< /dev/urandom tr -dc 'a-z0-9' | fold -w $$LENGTH | head -n 1); \
	IMAGE_NAME="ablator-local-test-$$RANDOM_STRING"; \
	CONTAINER_NAME="ablator-local-test-$$RANDOM_STRING"; \
	if [ ! -f $$DOCKERFILE ]; then echo "Dockerfile not found: $$DOCKERFILE"; exit 1; fi; \
	echo DOCKERFILE = $$DOCKERFILE, IMAGE_NAME = $$IMAGE_NAME, BUILD_CONTEXT = $$BUILD_CONTEXT;\
	docker build -f $$DOCKERFILE -t $$IMAGE_NAME $$BUILD_CONTEXT ; \
	docker run -it --rm --name $$CONTAINER_NAME $$IMAGE_NAME ; \
	docker rmi "$$IMAGE_NAME"


