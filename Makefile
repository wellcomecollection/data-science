.PHONY: setup lint check_version test build publish clean

default: build

# commands for building the package
setup:
	pip install flit
	flit install -s

lint:
	isort weco_datascience/*.py
	black weco_datascience/ --line-length 80
	flake8 weco_datascience/ --max-line-length 80

test:
	python -m pytest ./weco_datascience/test

show_versions:
	python --version
	pip freeze

build: show_versions clean lint test
	flit build


MODULE_VERSION := $(shell python -c "from weco_datascience import __version__ as v; print(v)")
check_version:
	[ "refs/tags/${MODULE_VERSION}" = "${GITHUB_REF}" ]

publish: check_version build
	flit publish

# general commands
clean:
	isort weco_datascience/**/*.py
	black weco_datascience --line-length 80
	flake8 weco_datascience --max-line-length 80 --ignore=E501,W291
	rm -rf *.pyc **/*.pyc
	rm -rf .hypothesis **/.hypothesis
	rm -rf .pytest_cache **/.pytest_cache
	rm -rf __pycache__ ./**/__pycache__
