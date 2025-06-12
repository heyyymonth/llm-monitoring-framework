.PHONY: help build test run clean up down logs shell

# Default target
help:
	@echo "LLM Quality & Safety Monitor - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  build       Build Docker image"
	@echo "  run         Run application with Docker"
	@echo "  run-api     Run API server only"
	@echo "  run-dash    Run dashboard only"
	@echo "  run-full    Run full stack (API + Dashboard)"
	@echo "  up          Start services with docker-compose"
	@echo "  down        Stop and remove containers"
	@echo "  logs        Show container logs"
	@echo "  shell       Get shell access to container"
	@echo "  clean       Remove all containers and images"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  test              Run tests in Docker container"
	@echo "  test-local        Run tests locally (no Docker)"
	@echo "  test-all-quality  Run ALL comprehensive quality tests"
	@echo "  test-quality      Run quality monitoring tests"
	@echo "  test-safety       Run safety monitoring tests"
	@echo "  test-cost         Run cost tracking tests"
	@echo "  test-benchmarks   Run performance benchmarks"
	@echo "  test-integration  Run integration tests"
	@echo "  test-fast         Run fast development tests"
	@echo "  test-regression   Run regression tests"
	@echo ""
	@echo "📊 Coverage & Analysis:"
	@echo "  test-quality-coverage  Run quality tests with coverage"
	@echo "  test-safety-coverage   Run safety tests with coverage" 
	@echo "  test-cost-coverage     Run cost tests with coverage"
	@echo "  benchmark              Run performance benchmarks"
	@echo ""
	@echo "🚀 Development Workflows:"
	@echo "  dev-test      Build and test (Docker)"
	@echo "  dev-quality   Run quality-focused test suite"

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
	python3 -m pytest tests/ -v

# Comprehensive quality testing commands
test-quality:
	@echo "🔍 Running comprehensive quality monitoring tests..."
	python3 -m pytest tests/test_comprehensive_quality.py -v

test-safety:
	@echo "🛡️ Running comprehensive safety monitoring tests..."
	python3 -m pytest tests/test_safety_comprehensive.py -v

test-benchmarks:
	@echo "📊 Running quality benchmarks and performance tests..."
	python3 -m pytest tests/test_quality_benchmarks.py -v -s

test-cost:
	@echo "💰 Running cost tracking tests..."
	python3 -m pytest tests/test_cost_tracking.py -v

test-integration:
	@echo "🔗 Running integration tests..."
	@echo "Note: Integration tests require Ollama and monitoring API running"
	python3 tests/test_integration_ollama.py

# Run all quality framework tests
test-all-quality:
	@echo "🎯 Running ALL quality framework tests..."
	@echo "Testing Quality Dimensions..."
	python3 -m pytest tests/test_comprehensive_quality.py -v
	@echo "Testing Safety Monitoring..."
	python3 -m pytest tests/test_safety_comprehensive.py -v
	@echo "Testing Cost Tracking..."
	python3 -m pytest tests/test_cost_tracking.py -v
	@echo "Testing Performance Benchmarks..."
	python3 -m pytest tests/test_quality_benchmarks.py -v -s
	@echo "✅ All quality framework tests complete!"
	@echo "💡 Run 'make test-integration' for manual integration testing"

# Run specific test categories with coverage
test-quality-coverage:
	@echo "🔍 Running quality tests with coverage..."
	python3 -m pytest tests/test_comprehensive_quality.py --cov=monitoring.quality --cov-report=term-missing -v

test-safety-coverage:
	@echo "🛡️ Running safety tests with coverage..."
	python3 -m pytest tests/test_safety_comprehensive.py --cov=monitoring.quality --cov-report=term-missing -v

test-cost-coverage:
	@echo "💰 Running cost tests with coverage..."
	python3 -m pytest tests/test_cost_tracking.py --cov=monitoring.cost --cov-report=term-missing -v

# Fast subset of tests for development
test-fast:
	@echo "⚡ Running fast development tests..."
	python3 -m pytest tests/test_quality_monitoring.py::TestQualityMonitor::test_evaluate_response_basic -v
	python3 -m pytest tests/test_comprehensive_quality.py::TestSemanticSimilarity::test_high_semantic_similarity_scenarios -v
	python3 -m pytest tests/test_safety_comprehensive.py::TestHallucinationDetection::test_explicit_hallucination_patterns -v

# Regression tests for quality metrics
test-regression:
	@echo "🔄 Running regression tests for quality metrics..."
	python3 -m pytest tests/test_quality_benchmarks.py::TestQualityRegressionTests -v

# Performance benchmarking
benchmark:
	@echo "🏃 Running performance benchmarks..."
	python3 -m pytest tests/test_quality_benchmarks.py::TestQualityBenchmarks::test_performance_benchmarks -v -s
	python3 -m pytest tests/test_quality_benchmarks.py::TestQualityBenchmarks::test_baseline_quality_scores -v -s

# Development helper - build and test
dev-test: build test
	@echo "✅ Build and test complete"

# Quality-focused development workflow
dev-quality: test-quality test-safety test-benchmarks
	@echo "✅ Quality-focused development tests complete" 