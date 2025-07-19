# fMRI-fNIRS Transfer Learning Pipeline
# Makefile for development and deployment

.PHONY: help install install-dev test test-coverage lint format clean colab-prep docker-build docker-run

# Default target
help:
	@echo "🧠 fMRI-fNIRS Transfer Learning Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  install         Install package in production mode"
	@echo "  install-dev     Install package in development mode with dev dependencies"
	@echo "  test           Run test suite"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  lint           Run code linting (flake8, mypy)"
	@echo "  format         Format code with black and isort"
	@echo "  clean          Clean build artifacts and cache"
	@echo "  colab-prep     Prepare package for Google Colab"
	@echo "  docker-build   Build Docker container"
	@echo "  docker-run     Run training in Docker"
	@echo ""
	@echo "🚀 Quick start:"
	@echo "  make install-dev && make test"

# Installation targets
install:
	@echo "📦 Installing fNIRS-Vise..."
	pip install -e .

install-dev:
	@echo "🛠️ Installing fNIRS-Vise in development mode..."
	pip install -e ".[dev,test]"

# Testing targets
test:
	@echo "🧪 Running test suite..."
	pytest tests/ -v --tb=short

test-coverage:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Code quality targets
lint:
	@echo "🔍 Running linting checks..."
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting passed!"

format:
	@echo "✨ Formatting code..."
	black src/ tests/ scripts/ --line-length 88
	isort src/ tests/ scripts/ --profile black
	@echo "✅ Code formatted!"

# Cleanup targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

# Google Colab preparation
colab-prep:
	@echo "📱 Preparing package for Google Colab..."
	@# Create a clean distribution
	python setup.py sdist bdist_wheel
	@# Create Colab-ready zip
	@echo "Creating Colab package..."
	zip -r colab-package.zip \
		src/ \
		configs/ \
		notebooks/ \
		scripts/ \
		requirements.txt \
		pyproject.toml \
		README.md \
		-x "*/__pycache__/*" "*.pyc" ".git/*"
	@echo "📦 Colab package ready: colab-package.zip"
	@echo "Upload this to Colab and run: !unzip colab-package.zip"

# Docker targets
docker-build:
	@echo "🐳 Building Docker container..."
	docker build -t fnirs-vise:latest .

docker-run:
	@echo "🚀 Running training in Docker..."
	docker run --gpus all -v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		fnirs-vise:latest \
		python -m src.train --config configs/docker.yaml

# Training shortcuts
train-colab:
	@echo "🧠 Starting Colab training..."
	python -m src.train --config configs/colab.yaml

train-local:
	@echo "🖥️ Starting local training..."
	python -m src.train --config configs/base.yaml

train-sweep:
	@echo "🔄 Starting hyperparameter sweep..."
	python -m src.train --multirun \
		transfer.mode=feature_guided,distill,concat \
		training.learning_rate=1e-4,3e-4,1e-3 \
		training.batch_size=8,16,32

# Data management
download-data:
	@echo "📥 Downloading sample data from OSF..."
	python scripts/download_data.py --project-id your_osf_id

prepare-data:
	@echo "🔄 Preprocessing raw data..."
	python scripts/preprocess.py --input-dir ./data/raw --output-dir ./data/processed

# Model evaluation
evaluate:
	@echo "📊 Evaluating trained models..."
	python scripts/evaluate.py --mode batch

demo:
	@echo "🎮 Running interactive demo..."
	python demo.py

# Development utilities
notebook:
	@echo "📓 Starting Jupyter server..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard:
	@echo "📈 Starting TensorBoard..."
	tensorboard --logdir=./lightning_logs --host=0.0.0.0 --port=6006

# CI/CD helpers
ci-test:
	@echo "🤖 Running CI tests..."
	pytest tests/ -v --junitxml=test-results.xml --cov=src --cov-report=xml

ci-build:
	@echo "🔨 Building for CI..."
	python -m build

# Documentation
docs-build:
	@echo "📚 Building documentation..."
	cd docs && make html

docs-serve:
	@echo "🌐 Serving documentation..."
	cd docs/_build/html && python -m http.server 8000

# Profiling and debugging
profile:
	@echo "⚡ Profiling training performance..."
	python -m cProfile -o profile.stats -m src.train --config configs/profile.yaml
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

debug:
	@echo "🐛 Starting debugging session..."
	python -m pdb -m src.train --config configs/debug.yaml

# Environment setup
setup-conda:
	@echo "🐍 Setting up Conda environment..."
	conda create -n fnirs-vise python=3.10 -y
	conda activate fnirs-vise
	$(MAKE) install-dev

setup-venv:
	@echo "🐍 Setting up virtual environment..."
	python -m venv venv
	source venv/bin/activate && $(MAKE) install-dev

# Performance benchmarks
benchmark:
	@echo "⏱️ Running performance benchmarks..."
	python scripts/benchmark.py --config configs/benchmark.yaml

memory-test:
	@echo "🧠 Testing memory usage..."
	python scripts/memory_test.py --max-batch-size 64
