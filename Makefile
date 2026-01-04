# MIVAA PDF Extractor - Test and Development Commands
# This Makefile provides convenient commands for testing, development, and maintenance

.PHONY: help test test-unit test-integration test-e2e test-performance test-security test-all
.PHONY: coverage coverage-report coverage-html clean install dev lint format check
.PHONY: setup-test-env run-server docs build

# Default target
help:
	@echo "MIVAA PDF Extractor - Available Commands:"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test-unit          Run unit tests with coverage"
	@echo "  test-integration   Run integration tests"
	@echo "  test-e2e          Run end-to-end tests"
	@echo "  test-performance  Run performance tests"
	@echo "  test-security     Run security tests"
	@echo "  test-all          Run all test suites"
	@echo "  test              Alias for test-unit"
	@echo ""
	@echo "Coverage Commands:"
	@echo "  coverage          Generate coverage report"
	@echo "  coverage-report   Generate and display coverage report"
	@echo "  coverage-html     Generate HTML coverage report"
	@echo ""
	@echo "Development Commands:"
	@echo "  install           Install dependencies"
	@echo "  dev               Start development server"
	@echo "  run-server        Start production server"
	@echo "  lint              Run code linting"
	@echo "  format            Format code"
	@echo "  check             Run all code quality checks"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean             Clean up generated files"
	@echo "  setup-test-env    Set up test environment"
	@echo "  docs              Generate documentation"
	@echo "  build             Build the application"

# Testing Commands
# NOTE: Test infrastructure is not yet implemented (tests/ directory is empty)
# Uncomment and use these targets once tests are written

# test: test-unit
#
# test-unit:
# 	@echo "🧪 Running Unit Tests..."
# 	python -m pytest tests/unit/ -v
#
# test-integration:
# 	@echo "🔗 Running Integration Tests..."
# 	python -m pytest tests/integration/ -v
#
# test-e2e:
# 	@echo "🌐 Running End-to-End Tests..."
# 	python -m pytest tests/e2e/ -v
#
# test-all:
# 	@echo "🚀 Running All Tests..."
# 	python -m pytest tests/ -v
#
# # Coverage Commands
# coverage:
# 	@echo "📊 Generating Coverage Report..."
# 	python -m pytest tests/unit/ --cov=app --cov-report=term-missing
#
# coverage-report:
# 	@echo "📋 Generating Detailed Coverage Report..."
# 	python -m pytest tests/unit/ --cov=app --cov-report=term-missing --cov-report=xml
# 	@echo "Coverage report generated: coverage.xml"

# coverage-html:
# 	@echo "🌐 Generating HTML Coverage Report..."
# 	python -m pytest tests/unit/ --cov=app --cov-report=html
# 	@echo "HTML coverage report generated: htmlcov/index.html"

# Development Commands
install:
	@echo "📦 Installing Dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

dev:
	@echo "🚀 Starting Development Server..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-server:
	@echo "🏭 Starting Production Server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000

lint:
	@echo "🔍 Running Code Linting..."
	flake8 app/ tests/
	pylint app/ tests/

format:
	@echo "✨ Formatting Code..."
	black app/ tests/
	isort app/ tests/

check: lint
	@echo "✅ Running Code Quality Checks..."
	mypy app/
	bandit -r app/

# Maintenance Commands
clean:
	@echo "🧹 Cleaning Up..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf test_reports/
	rm -f coverage.xml
	rm -f .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

setup-test-env:
	@echo "🔧 Setting Up Test Environment..."
	mkdir -p test_reports
	mkdir -p htmlcov
	mkdir -p logs
	@echo "Test environment ready!"

docs:
	@echo "📚 Generating Documentation..."
	@echo "Documentation generation not yet implemented"

build:
	@echo "🏗️ Building Application..."
	@echo "Build process not yet implemented"

# Quick test commands for development (commented out until tests are implemented)
# quick-test:
# 	@echo "⚡ Running Quick Unit Tests..."
# 	python -m pytest tests/unit/ -x -v --tb=short
#
# watch-test:
# 	@echo "👀 Running Tests in Watch Mode..."
# 	python -m pytest tests/unit/ -f
#
# # Parallel testing
# test-parallel:
# 	@echo "🚀 Running Tests in Parallel..."
# 	python -m pytest tests/ -n auto
#
# # Full quality check
# quality-check: lint
# 	@echo "✅ Code Quality Check Complete!"