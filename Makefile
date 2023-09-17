.PHONY: test
test:
	bash scripts/run_tests.sh

test-cpu:
	bash scripts/run_tests.sh --cpu

install:
	pip install -e ."[dev]"

static-checks:
	bash scripts/run_lint.sh
