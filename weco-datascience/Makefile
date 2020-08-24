.PHONY: setup lint test build publish clean

default: build

# commands for building the package
setup:
	pip install flit
	flit install -s

lint:
	isort weco_datascience/*.py
	black weco_datascience/ --line-length 80
	flake8 weco_datascience/

test:
	python -m unittest discover -s ./test

build: clean lint test
	flit build

publish: build
	flit publish

# general commands
clean:
	rm -rf .hypothesis
	rm -rf .pytest_cache
	rm -rf ./*/__pycache__
	rm -rf ./dist
	rm -rf ./site
