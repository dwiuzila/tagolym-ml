SHELL = /bin/bash

# Set variable
PROJECT_PATH := "."

# Help
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."

# Styling
.PHONY: style
style:
	black .
	isort .
	flake8

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -rf .coverage*

# Environment
.PHONY: venv
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install --upgrade pip && \
	python3 -m pip install -e ${PROJECT_PATH}