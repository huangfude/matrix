# Matrix Router Makefile

.PHONY: help build wheel test install uninstall clean format docs

# Default target
help:
	@echo "Matrix Router - Available commands:"
	@echo "  make build     - Build the Rust project"
	@echo "  make test      - Run tests"
	@echo "  make install   - Install Python package"
	@echo "  make uninstall   - Uninstall Python package"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make format    - Format code"
	@echo "  make docs      - Documentation"

# Build targets
build:
	cargo build -p matrix_router

# Build wheel
# uv build --wheel --out-dir ./dist
wheel:
	python3 -m build ./matrix_cli --wheel --outdir ./dist

# Test targets
test:
	cargo test

# Install targets
# uv pip install -e .
install:
	pip install ./dist/matrix_cli-*.whl

uninstall:
	pip uninstall matrix-cli -y

# Clean targets
clean:
	cargo clean
	rm -rf dist
	rm -rf target
	rm -rf *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Formatting and linting
format:
	cargo fmt

# Documentation
docs:
	cargo doc --open

