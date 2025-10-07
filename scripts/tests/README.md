# MIVAA PDF Extractor Tests

Test scripts specific to the MIVAA PDF Extractor service.

## Files

### Python Test Scripts
- `test_embedding_migration.py` - Tests embedding migration functionality
- `test_embedding_validation.py` - Validates embedding generation
- `test_enhanced_extraction.py` - Tests enhanced PDF extraction features
- `test_image_integration.py` - Tests image processing integration

### Deployment Testing
- `test-deployment.sh` - Shell script for testing deployment

## Usage

```bash
# Run Python tests
cd mivaa-pdf-extractor
python scripts/tests/test_embedding_migration.py
python scripts/tests/test_embedding_validation.py
python scripts/tests/test_enhanced_extraction.py
python scripts/tests/test_image_integration.py

# Run deployment test
bash scripts/tests/test-deployment.sh
```

## Purpose

These scripts test:
1. PDF extraction and processing functionality
2. Embedding generation and migration
3. Image processing integration
4. Deployment validation
5. Service-specific features

## Requirements

- Python 3.8+
- Required dependencies from `requirements.txt`
- Proper environment configuration
- Access to test PDF files
