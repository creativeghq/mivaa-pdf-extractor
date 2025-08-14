#!/bin/bash

# MIVAA PDF Extractor - Digital Ocean Deployment Script
# This script automates the deployment process to Digital Ocean droplets

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/mivaa-deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

warning() {
    log "${YELLOW}WARNING: ${1}${NC}"
}

info() {
    log "${BLUE}INFO: ${1}${NC}"
}

success() {
    log "${GREEN}SUCCESS: ${1}${NC}"
}

# Help function
show_help() {
    cat << EOF
MIVAA PDF Extractor - Digital Ocean Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -e, --environment ENV   Target environment (production, staging) [default: production]
    -v, --validate-only     Only validate configuration without deploying
    -f, --force             Force deployment without confirmation
    --skip-tests           Skip running tests before deployment
    --skip-build           Skip building Docker image
    --dry-run              Show what would be deployed without executing

EXAMPLES:
    $0                                    # Deploy to production with confirmation
    $0 -e staging                        # Deploy to staging environment
    $0 --validate-only                   # Only validate configuration
    $0 --force --skip-tests              # Force deploy without tests
    $0 --dry-run                         # Show deployment plan

ENVIRONMENT VARIABLES:
    Required for deployment:
    - DEPLOY_HOST: Target server hostname/IP
    - DEPLOY_USER: SSH username for deployment
    - DEPLOY_SSH_KEY: Path to SSH private key
    - GITHUB_TOKEN: GitHub token for container registry

    Application secrets (validated):
    - SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
    - OPENAI_API_KEY
    - JWT_SECRET_KEY
    - MATERIAL_KAI_API_KEY
    - ENCRYPTION_KEY

EOF
}

# Parse command line arguments
ENVIRONMENT="production"
VALIDATE_ONLY=false
FORCE_DEPLOY=false
SKIP_TESTS=false
SKIP_BUILD=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_DEPLOY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(production|staging)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be 'production' or 'staging'."
fi

info "Starting MIVAA PDF Extractor deployment to $ENVIRONMENT environment"
info "Log file: $LOG_FILE"

# Check required tools
check_dependencies() {
    info "Checking dependencies..."
    
    local missing_tools=()
    
    for tool in docker git ssh python3; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
    fi
    
    success "All required tools are available"
}

# Validate deployment configuration
validate_deployment_config() {
    info "Validating deployment configuration..."
    
    # Check deployment environment variables
    local required_deploy_vars=(
        "DEPLOY_HOST"
        "DEPLOY_USER" 
        "DEPLOY_SSH_KEY"
        "GITHUB_TOKEN"
    )
    
    local missing_vars=()
    
    for var in "${required_deploy_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required deployment environment variables: ${missing_vars[*]}"
    fi
    
    # Validate SSH key
    if [[ ! -f "$DEPLOY_SSH_KEY" ]]; then
        error "SSH key file not found: $DEPLOY_SSH_KEY"
    fi
    
    # Test SSH connection
    info "Testing SSH connection to $DEPLOY_HOST..."
    if ! ssh -i "$DEPLOY_SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$DEPLOY_USER@$DEPLOY_HOST" "echo 'SSH connection successful'" &>/dev/null; then
        error "Failed to connect to $DEPLOY_HOST via SSH"
    fi
    
    success "Deployment configuration is valid"
}

# Validate application secrets using Python script
validate_application_secrets() {
    info "Validating application secrets..."
    
    if [[ -f "$PROJECT_ROOT/scripts/validate-deployment.py" ]]; then
        if ! python3 "$PROJECT_ROOT/scripts/validate-deployment.py" --environment="$ENVIRONMENT"; then
            error "Application secrets validation failed"
        fi
        success "Application secrets validation passed"
    else
        warning "Validation script not found, skipping application secrets validation"
    fi
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warning "Skipping tests as requested"
        return 0
    fi
    
    info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies if needed
    if [[ -f "requirements.txt" ]]; then
        python3 -m pip install -r requirements.txt --quiet
    fi
    
    # Run tests
    if [[ -f "run_tests.py" ]]; then
        if ! python3 run_tests.py; then
            error "Tests failed"
        fi
    else
        warning "Test runner not found, skipping tests"
    fi
    
    success "Tests passed"
}

# Build Docker image
build_docker_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        warning "Skipping Docker build as requested"
        return 0
    fi
    
    info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    local image_tag="ghcr.io/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/mivaa-pdf-extractor:$(git rev-parse --short HEAD)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would build image: $image_tag"
        return 0
    fi
    
    # Build image
    if ! docker build -t "$image_tag" .; then
        error "Docker build failed"
    fi
    
    # Login to GitHub Container Registry
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$(git config --get user.name)" --password-stdin
    
    # Push image
    if ! docker push "$image_tag"; then
        error "Docker push failed"
    fi
    
    success "Docker image built and pushed: $image_tag"
    echo "$image_tag" > /tmp/mivaa-image-tag
}

# Deploy to server
deploy_to_server() {
    info "Deploying to server..."
    
    local image_tag
    if [[ -f "/tmp/mivaa-image-tag" ]]; then
        image_tag=$(cat /tmp/mivaa-image-tag)
    else
        image_tag="ghcr.io/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')/mivaa-pdf-extractor:$(git rev-parse --short HEAD)"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would deploy image: $image_tag"
        info "DRY RUN: Would update environment variables on $DEPLOY_HOST"
        return 0
    fi
    
    # Create deployment script on server
    ssh -i "$DEPLOY_SSH_KEY" "$DEPLOY_USER@$DEPLOY_HOST" << EOF
set -euo pipefail

echo "🚀 Starting deployment on \$(hostname)"

# Navigate to application directory
cd /opt/mivaa-pdf-extractor || {
    echo "❌ Application directory not found. Please run server setup first."
    exit 1
}

# Pull latest code
git fetch origin
git reset --hard origin/main

# Login to GitHub Container Registry
echo "$GITHUB_TOKEN" | docker login ghcr.io -u \$(git config --get user.name) --password-stdin

# Create docker-compose override for $ENVIRONMENT
cat > docker-compose.override.yml << 'OVERRIDE_EOF'
version: '3.8'
services:
  mivaa-pdf-extractor:
    image: $image_tag
    environment:
      # Core Application
      - ENVIRONMENT=$ENVIRONMENT
      - DEBUG=false
      - LOG_LEVEL=INFO
      - APP_NAME=MIVAA PDF Extractor
      - APP_VERSION=1.0.0
      
      # Database Configuration
      - SUPABASE_URL=\${SUPABASE_URL}
      - SUPABASE_ANON_KEY=\${SUPABASE_ANON_KEY}
      - SUPABASE_SERVICE_ROLE_KEY=\${SUPABASE_SERVICE_ROLE_KEY}
      - DATABASE_URL=\${DATABASE_URL:-}
      - DB_POOL_SIZE=\${DB_POOL_SIZE:-10}
      - DB_MAX_OVERFLOW=\${DB_MAX_OVERFLOW:-20}
      - DB_POOL_TIMEOUT=\${DB_POOL_TIMEOUT:-30}
      
      # AI/ML Service API Keys
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - OPENAI_ORG_ID=\${OPENAI_ORG_ID:-}
      - OPENAI_PROJECT_ID=\${OPENAI_PROJECT_ID:-}
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY:-}
      - HUGGINGFACE_API_TOKEN=\${HUGGINGFACE_API_TOKEN:-}
      - HUGGING_FACE_ACCESS_TOKEN=\${HUGGING_FACE_ACCESS_TOKEN:-}
      - REPLICATE_API_KEY=\${REPLICATE_API_KEY:-}
      - JINA_API_KEY=\${JINA_API_KEY:-}
      - FIRECRAWL_API_KEY=\${FIRECRAWL_API_KEY:-}
      
      # JWT Authentication
      - JWT_SECRET_KEY=\${JWT_SECRET_KEY}
      - JWT_ALGORITHM=\${JWT_ALGORITHM:-HS256}
      - JWT_ACCESS_TOKEN_EXPIRE_MINUTES=\${JWT_ACCESS_TOKEN_EXPIRE_MINUTES:-30}
      - JWT_REFRESH_TOKEN_EXPIRE_DAYS=\${JWT_REFRESH_TOKEN_EXPIRE_DAYS:-7}
      - JWT_ISSUER=\${JWT_ISSUER:-mivaa-pdf-extractor}
      - JWT_AUDIENCE=\${JWT_AUDIENCE:-mivaa-users}
      
      # Error Tracking & Monitoring
      - SENTRY_DSN=\${SENTRY_DSN:-}
      - SENTRY_AUTH_TOKEN=\${SENTRY_AUTH_TOKEN:-}
      - SENTRY_ORG=\${SENTRY_ORG:-}
      - SENTRY_PROJECT=\${SENTRY_PROJECT:-}
      - SENTRY_ENVIRONMENT=$ENVIRONMENT
      - SENTRY_TRACES_SAMPLE_RATE=\${SENTRY_TRACES_SAMPLE_RATE:-0.1}
      - SENTRY_PROFILES_SAMPLE_RATE=\${SENTRY_PROFILES_SAMPLE_RATE:-0.1}
      
      # Material Kai Platform Integration
      - MATERIAL_KAI_API_URL=\${MATERIAL_KAI_API_URL:-}
      - MATERIAL_KAI_API_KEY=\${MATERIAL_KAI_API_KEY:-}
      - MATERIAL_KAI_CLIENT_ID=\${MATERIAL_KAI_CLIENT_ID:-}
      - MATERIAL_KAI_CLIENT_SECRET=\${MATERIAL_KAI_CLIENT_SECRET:-}
      - MATERIAL_KAI_WEBHOOK_SECRET=\${MATERIAL_KAI_WEBHOOK_SECRET:-}
      
      # Performance & Caching
      - MAX_WORKERS=\${MAX_WORKERS:-4}
      - WORKER_TIMEOUT=\${WORKER_TIMEOUT:-300}
      - MAX_UPLOAD_SIZE=\${MAX_UPLOAD_SIZE:-100}
      
      # Security & CORS
      - CORS_ORIGINS=\${CORS_ORIGINS:-*}
      - CORS_ALLOW_CREDENTIALS=\${CORS_ALLOW_CREDENTIALS:-true}
      - RATE_LIMIT_PER_MINUTE=\${RATE_LIMIT_PER_MINUTE:-60}
      - RATE_LIMIT_BURST=\${RATE_LIMIT_BURST:-10}
      - ENCRYPTION_KEY=\${ENCRYPTION_KEY}
      
      # CI/CD Pipeline Variables
      - GITHUB_TOKEN=\${GITHUB_TOKEN}
      - DOCKER_REGISTRY_URL=\${DOCKER_REGISTRY_URL:-ghcr.io}
      - DOCKER_REGISTRY_USERNAME=\${DOCKER_REGISTRY_USERNAME:-\$(git config --get user.name)}
      - DOCKER_REGISTRY_PASSWORD=\${DOCKER_REGISTRY_PASSWORD:-\$GITHUB_TOKEN}
OVERRIDE_EOF

# Pull latest image
docker pull $image_tag

# Stop existing containers
docker-compose down

# Start new containers with override
docker-compose up -d

# Clean up old images
docker image prune -f

# Wait for health check
echo "⏳ Waiting for application to be healthy..."
for i in {1..30}; do
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ Application is healthy!"
        break
    fi
    echo "Attempt \$i/30: Application not ready yet..."
    sleep 10
done

# Verify deployment
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Deployment successful!"
    echo "🌐 Application is running at: http://\$(hostname)"
else
    echo "❌ Health check failed!"
    docker-compose logs --tail=50
    exit 1
fi
EOF
    
    success "Deployment completed successfully"
}

# Confirmation prompt
confirm_deployment() {
    if [[ "$FORCE_DEPLOY" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    echo
    warning "You are about to deploy MIVAA PDF Extractor to $ENVIRONMENT environment"
    warning "Target server: $DEPLOY_HOST"
    echo
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Deployment cancelled by user"
        exit 0
    fi
}

# Main deployment flow
main() {
    check_dependencies
    validate_deployment_config
    validate_application_secrets
    
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        success "Validation completed successfully"
        exit 0
    fi
    
    confirm_deployment
    
    run_tests
    build_docker_image
    deploy_to_server
    
    success "🎉 MIVAA PDF Extractor deployment to $ENVIRONMENT completed successfully!"
    info "📋 Deployment log: $LOG_FILE"
}

# Run main function
main "$@"