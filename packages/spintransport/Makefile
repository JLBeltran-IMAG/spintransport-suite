# Package Makefile — setuptools backend via 'python -m build'
# Works with pyproject.toml (build-backend = "setuptools.build_meta")

PY        ?= python3
PIP       ?= pip
DIST_DIR  := dist

.PHONY: help build wheel sdist install dev check clean publish-test publish

help:
	@echo "Package targets:"
	@echo "  build        - python -m build (wheel + sdist)"
	@echo "  wheel        - python -m build --wheel"
	@echo "  sdist        - python -m build --sdist"
	@echo "  install      - pip install --force-reinstall --no-deps dist/*.whl"
	@echo "  dev          - pip install -e .[dev]  (PEP 660 editable)"
	@echo "  check        - twine check dist/*"
	@echo "  publish-test - twine upload -r testpypi dist/*"
	@echo "  publish      - twine upload dist/*"
	@echo "  clean        - remove build artifacts"

build:
	$(PY) -m pip install -q --upgrade pip build
	$(PY) -m build

wheel:
	$(PY) -m pip install -q --upgrade pip build
	$(PY) -m build --wheel

sdist:
	$(PY) -m pip install -q --upgrade pip build
	$(PY) -m build --sdist

install: wheel
	$(PIP) install --force-reinstall --no-deps $(DIST_DIR)/*.whl

dev:
	$(PIP) install -e .[dev]

check:
	$(PY) -m pip install -q twine
	$(PY) -m twine check $(DIST_DIR)/*

publish-test: check
	$(PY) -m twine upload -r testpypi $(DIST_DIR)/*

publish: check
	$(PY) -m twine upload $(DIST_DIR)/*

clean:
	rm -rf build $(DIST_DIR) *.egg-info
