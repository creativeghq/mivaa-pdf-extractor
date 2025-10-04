#!/bin/bash

# MIVAA PDF Extractor - OpenAPI Schema Update Script
# This script updates the OpenAPI schema files for deployment

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if service is running
check_service() {
    local service_url="${1:-http://localhost:8000}"
    local max_attempts=30
    local attempt=1
    
    log "🔍 Checking if MIVAA service is running at $service_url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$service_url/health" > /dev/null 2>&1; then
            log "✅ Service is running and healthy"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Service not responding after $max_attempts attempts"
            error "Please ensure the MIVAA service is running at $service_url"
            return 1
        fi
        
        log "⏳ Waiting for service (attempt $attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done
}

# Check if service is running with comprehensive API
check_comprehensive_api() {
    local service_url="${1:-http://localhost:8000}"

    log "🔍 Checking if service is running with comprehensive API..."

    if curl -s "$service_url/health" > /dev/null 2>&1; then
        # Check if it's the comprehensive API
        local endpoint_count=$(curl -s "$service_url/" 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('api_info', {}).get('total_endpoints', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

        if [ "$endpoint_count" -gt 10 ]; then
            log "✅ Service is running with comprehensive API ($endpoint_count endpoints)"
            return 0
        else
            warn "Service is running but with legacy API (only $endpoint_count endpoints)"
            warn "Manual restart may be needed to enable comprehensive API"
            warn "Use: docker-compose restart or uvicorn app.main:app --host 0.0.0.0 --port 8000"
            return 1
        fi
    else
        error "Service is not responding"
        return 1
    fi
}

# Generate OpenAPI schema
generate_schema() {
    local service_url="${1:-http://localhost:8000}"
    
    log "🔄 Generating OpenAPI schema from $service_url..."
    
    # Create output directory
    mkdir -p docs/openapi
    
    # Download OpenAPI JSON
    log "📄 Downloading OpenAPI JSON schema..."
    if curl -s "$service_url/openapi.json" > docs/openapi/openapi.json; then
        log "✅ OpenAPI JSON downloaded successfully"
    else
        error "Failed to download OpenAPI JSON from $service_url/openapi.json"
        return 1
    fi
    
    # Validate JSON
    if ! python -m json.tool docs/openapi/openapi.json > /dev/null 2>&1; then
        error "Downloaded OpenAPI JSON is not valid"
        return 1
    fi
    
    # Generate JavaScript module
    log "📄 Generating JavaScript module..."
    cat > docs/openapi/openapi.js << EOF
// MIVAA PDF Extractor - OpenAPI Schema
// Auto-generated from FastAPI application
// Generated on: $(date)
// Source: $service_url/openapi.json

export const openApiSchema = $(cat docs/openapi/openapi.json);

export default openApiSchema;
EOF
    
    # Generate TypeScript module
    log "📄 Generating TypeScript module..."
    cat > docs/openapi/openapi.ts << 'EOF'
// MIVAA PDF Extractor - OpenAPI Schema
// Auto-generated from FastAPI application
// Generated on: $(date)
// Source: $service_url/openapi.json

export interface OpenAPISchema {
  openapi: string;
  info: {
    title: string;
    version: string;
    description?: string;
    [key: string]: any;
  };
  servers?: Array<{
    url: string;
    description?: string;
  }>;
  paths: { [key: string]: any };
  components?: { [key: string]: any };
  tags?: Array<{
    name: string;
    description?: string;
  }>;
  [key: string]: any;
}

export const openApiSchema: OpenAPISchema = 
EOF
    cat docs/openapi/openapi.json >> docs/openapi/openapi.ts
    echo ";" >> docs/openapi/openapi.ts
    echo "" >> docs/openapi/openapi.ts
    echo "export default openApiSchema;" >> docs/openapi/openapi.ts
    
    # Get schema statistics
    local endpoints=$(python -c "import json; data=json.load(open('docs/openapi/openapi.json')); print(len(data.get('paths', {})))" 2>/dev/null || echo "unknown")
    local version=$(python -c "import json; data=json.load(open('docs/openapi/openapi.json')); print(data.get('info', {}).get('version', 'unknown'))" 2>/dev/null || echo "unknown")
    local title=$(python -c "import json; data=json.load(open('docs/openapi/openapi.json')); print(data.get('info', {}).get('title', 'unknown'))" 2>/dev/null || echo "unknown")
    
    log "✅ OpenAPI schema generated successfully!"
    log "📁 Files created:"
    log "   - docs/openapi/openapi.json"
    log "   - docs/openapi/openapi.js"
    log "   - docs/openapi/openapi.ts"
    log "📊 Schema Statistics:"
    log "   - API Title: $title"
    log "   - API Version: $version"
    log "   - Total Endpoints: $endpoints"
}

# Main function
main() {
    local service_url="${1:-http://localhost:8000}"
    
    log "🚀 Starting OpenAPI schema update process..."
    log "🎯 Target service: $service_url"

    # Check if service is running
    if ! check_service "$service_url"; then
        error "Cannot proceed without a running service"
        exit 1
    fi

    # Check if service is running with comprehensive API (informational)
    if ! check_comprehensive_api "$service_url"; then
        warn "Service appears to be running with legacy API"
        warn "Schema will be generated from current running service"
        warn "For comprehensive API, manually restart with: docker-compose restart"
    fi

    # Generate schema
    if ! generate_schema "$service_url"; then
        error "Failed to generate OpenAPI schema"
        exit 1
    fi
    
    log "🎉 OpenAPI schema update completed successfully!"
    log "📍 Next steps:"
    log "   1. Commit the updated schema files to version control"
    log "   2. Deploy to update any external documentation"
    log "   3. Verify the schema at $service_url/docs"
}

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [SERVICE_URL]"
    echo ""
    echo "Updates OpenAPI schema files from a running MIVAA service."
    echo ""
    echo "Arguments:"
    echo "  SERVICE_URL    Base URL of the MIVAA service (default: http://localhost:8000)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default localhost:8000"
    echo "  $0 http://localhost:8000              # Explicit localhost"
    echo "  $0 https://your-domain.com            # Production service"
    echo ""
    echo "Generated files:"
    echo "  - docs/openapi/openapi.json           # OpenAPI JSON schema"
    echo "  - docs/openapi/openapi.js             # JavaScript module"
    echo "  - docs/openapi/openapi.ts             # TypeScript module"
    exit 0
fi

# Run main function
main "$@"
