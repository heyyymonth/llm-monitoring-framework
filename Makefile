.PHONY: help build test run clean up down logs shell

# Default target
help:
	@echo "LLM Quality & Safety Monitor - Docker Commands"
	@echo "=============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  build       Build Docker image"
	@echo "  test        Run tests in Docker container"
	@echo "  run         Run application with Docker"
	@echo "  run-api     Run API server only"
	@echo "  run-dash    Run dashboard only"
	@echo "  run-full    Run full stack (API + Dashboard)"
	@echo "  up          Start services with docker-compose"
	@echo "  down        Stop and remove containers"
	@echo "  logs        Show container logs"
	@echo "  shell       Get shell access to container"
	@echo "  clean       Remove all containers and images"
	@echo "  test-local  Run tests locally (no Docker)"

# Docker image name
IMAGE_NAME = llm-monitor
TAG = latest

# Build Docker image
build:
	@echo "🐳 Building Docker image..."
	docker build -t $(IMAGE_NAME):$(TAG) .

# Run tests in Docker container
test:
	@echo "🧪 Running tests in Docker container..."
	docker build -t $(IMAGE_NAME):test .
	docker run --rm $(IMAGE_NAME):test python -m pytest tests/ -v

# Run application with Docker
run:
	@echo "🚀 Running LLM Monitor in Docker..."
	docker run --rm -p 8000:8000 -p 8080:8080 $(IMAGE_NAME):$(TAG)

# Run API server only
run-api:
	@echo "🚀 Running API server only..."
	docker run --rm -p 8000:8000 $(IMAGE_NAME):$(TAG) python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Run dashboard only
run-dash:
	@echo "📊 Running dashboard only..."
	docker run --rm -p 8080:8080 $(IMAGE_NAME):$(TAG) python dashboard/app.py

# Run full stack with docker-compose
run-full:
	@echo "🚀 Running full stack with docker-compose..."
	docker-compose --profile full-stack up

# Start services with docker-compose
up:
	@echo "🚀 Starting services with docker-compose..."
	docker-compose up -d

# Stop and remove containers
down:
	@echo "🛑 Stopping containers..."
	docker-compose down

# Show container logs
logs:
	docker-compose logs -f

# Get shell access to container
shell:
	docker run --rm -it $(IMAGE_NAME):$(TAG) /bin/bash

# Clean up Docker resources
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Run tests locally (no Docker)
test-local:
	@echo "🧪 Running tests locally..."
	python -m pytest tests/ -v

# Development helper - build and test
dev-test: build test
	@echo "✅ Build and test complete" 