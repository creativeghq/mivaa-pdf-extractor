# MIVAA Scripts Directory

This directory contains operational scripts for the MIVAA PDF Extractor service.

## Directory Structure

```
scripts/
├── testing/              # E2E and integration testing scripts
├── database/             # Database management scripts
├── tests/                # Test utilities
├── clear_jobs_before_deployment.py
├── deployment_verification.py
└── validate-deployment-readiness.py
```

## Scripts Overview

### Production Scripts

#### `clear_jobs_before_deployment.py`
**Purpose**: Clear all pending/running background jobs before deployment  
**Usage**: `python scripts/clear_jobs_before_deployment.py`  
**When to use**: Before deploying to prevent stuck jobs from old code versions

#### `deployment_verification.py`
**Purpose**: Verify deployment health and service availability  
**Usage**: `python scripts/deployment_verification.py`  
**When to use**: After deployment to confirm all services are healthy

#### `validate-deployment-readiness.py`
**Purpose**: Pre-deployment validation checks  
**Usage**: `python scripts/validate-deployment-readiness.py`  
**When to use**: Before deployment to ensure environment is ready

### Testing Scripts (`testing/`)

#### `comprehensive_pdf_test.py`
**Purpose**: Complete E2E test for PDF processing pipeline  
**Usage**: `python scripts/testing/comprehensive_pdf_test.py`  
**Features**:
- Tests all 9 processing stages
- Reports 12 comprehensive metrics
- Validates KEDA autoscaling
- Checks database integrity

**Metrics Reported**:
1. Total Products discovered
2. Pages processed
3. Chunks created
4. Images processed
5. Embeddings created (5 types: text, CLIP, color, texture, application)
6. Errors encountered
7. Relationships created
8. Metadata extracted
9. Memory used
10. CPU used
11. Cost estimate
12. Total processing time

#### `monitor-job.sh`
**Purpose**: Monitor background job progress in real-time  
**Usage**: `./scripts/testing/monitor-job.sh <job_id>`

#### `run-fresh-test.sh`
**Purpose**: Run comprehensive test with fresh database state  
**Usage**: `./scripts/testing/run-fresh-test.sh`

#### `run_comprehensive_test.sh`
**Purpose**: Wrapper script for comprehensive PDF test  
**Usage**: `./scripts/testing/run_comprehensive_test.sh`

### Database Scripts (`database/`)

See `database/README.md` for database-specific scripts.

### Test Utilities (`tests/`)

See `tests/README.md` for test utility scripts.

## Deployment Workflow

1. **Pre-deployment**: Run `validate-deployment-readiness.py`
2. **Clear jobs**: Run `clear_jobs_before_deployment.py`
3. **Deploy**: GitHub Actions workflow handles deployment
4. **Verify**: Run `deployment_verification.py`
5. **E2E Test**: Run `comprehensive_pdf_test.py`

## Environment Variables

All scripts expect these environment variables:
- `MIVAA_API`: API endpoint (default: `https://v1api.materialshub.gr`)
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_ANON_KEY`: Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Supabase service role key

## Notes

- All testing scripts are designed to run from the repository root
- Scripts use production API by default (can be overridden with env vars)
- Comprehensive test takes ~4-5 minutes for a typical PDF
- KEDA autoscaling is validated during E2E tests

