.PHONY: setup lint check_version test build publish clean

default: build

# commands for building the package
setup:
	python -m pip install --upgrade pip
	pip install flit
	flit install -s

lint:
	isort weco_datascience/*.py
	black weco_datascience/ --line-length 80
	flake8 weco_datascience/ --max-line-length 80

test:
	python -m pytest ./weco_datascience/test

build: clean lint test
	flit build


MODULE_VERSION := $(shell python -c "from weco_datascience import __version__ as v; print(v)")
check_version:
	[ "refs/tags/${MODULE_VERSION}" = "${GITHUB_REF}" ]

publish: check_version build
	flit publish

# general commands
clean:
	rm -rf .hypothesis
	rm -rf .pytest_cache
	rm -rf ./*/__pycache__
	rm -rf ./dist
	rm -rf ./site
