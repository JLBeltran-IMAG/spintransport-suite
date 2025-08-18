# Root Makefile — orchestrates package build/install via the package Makefile

PKG_DIR := packages/spintransport

.PHONY: help build wheel sdist install dev check clean clean-all publish-test publish

help:
	@echo "Targets:"
	@echo "  build        - build wheel+sdist under $(PKG_DIR)/dist/"
	@echo "  wheel        - build wheel only"
	@echo "  sdist        - build sdist only"
	@echo "  install      - pip install wheel (force, no deps)"
	@echo "  dev          - pip install -e .[dev] (editable)"
	@echo "  check        - twine check dist artifacts"
	@echo "  publish-test - upload to TestPyPI (requires creds)"
	@echo "  publish      - upload to PyPI (requires creds)"
	@echo "  clean        - clean package build artifacts"
	@echo "  clean-all    - clean + remove __pycache__ across repo"

build:
	$(MAKE) -C $(PKG_DIR) build

wheel:
	$(MAKE) -C $(PKG_DIR) wheel

sdist:
	$(MAKE) -C $(PKG_DIR) sdist

install:
	$(MAKE) -C $(PKG_DIR) install

dev:
	$(MAKE) -C $(PKG_DIR) dev

check:
	$(MAKE) -C $(PKG_DIR) check

publish-test:
	$(MAKE) -C $(PKG_DIR) publish-test

publish:
	$(MAKE) -C $(PKG_DIR) publish

clean:
	$(MAKE) -C $(PKG_DIR) clean

clean-all: clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
