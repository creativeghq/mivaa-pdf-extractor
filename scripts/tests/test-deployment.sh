#!/bin/bash

# MIVAA PDF Extractor - Deployment Test
# This script tests that the comprehensive API is working properly

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Configuration
BASE_URL="${1:-http://localhost:8000}"
TIMEOUT=30

# Test function
test_endpoint() {
    local endpoint="$1"
    local expected_status="${2:-200}"
    local description="$3"
    
    info "Testing: $description"
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$BASE_URL$endpoint" || echo "000")
    
    if [ "$response_code" = "$expected_status" ]; then
        log "✅ $description - Status: $response_code"
        return 0
    else
        error "❌ $description - Expected: $expected_status, Got: $response_code"
        return 1
    fi
}

# Main test function
main() {
    log "🧪 Starting MIVAA API deployment tests..."
    log "🌐 Base URL: $BASE_URL"
    
    local failed_tests=0
    local total_tests=0
    
    # Core health endpoints
    info "=== Core Health Endpoints ==="
    
    ((total_tests++))
    if test_endpoint "/health" "200" "Health Check"; then
        :
    else
        ((failed_tests++))
    fi
    
    ((total_tests++))
    if test_endpoint "/docs" "200" "API Documentation"; then
        :
    else
        ((failed_tests++))
    fi
    
    ((total_tests++))
    if test_endpoint "/openapi.json" "200" "OpenAPI Schema"; then
        :
    else
        ((failed_tests++))
    fi
    
    # API v1 endpoints (require authentication, expect 401)
    info "=== API v1 Endpoints (Authentication Required) ==="
    
    ((total_tests++))
    if test_endpoint "/api/v1/health" "401" "API v1 Health (Auth Required)"; then
        :
    else
        ((failed_tests++))
    fi
    
    # RAG endpoints (require authentication, expect 401)
    info "=== RAG System Endpoints (Authentication Required) ==="
    
    ((total_tests++))
    if test_endpoint "/api/v1/rag/health" "401" "RAG Health (Auth Required)"; then
        :
    else
        ((failed_tests++))
    fi
    
    # Search endpoints (require authentication, expect 401)
    info "=== Search Endpoints (Authentication Required) ==="
    
    ((total_tests++))
    if test_endpoint "/api/search/health" "401" "Search Health (Auth Required)"; then
        :
    else
        ((failed_tests++))
    fi
    
    # Embedding endpoints (require authentication, expect 401)
    info "=== Embedding Endpoints (Authentication Required) ==="
    
    ((total_tests++))
    if test_endpoint "/api/embeddings/health" "401" "Embeddings Health (Auth Required)"; then
        :
    else
        ((failed_tests++))
    fi
    
    # Chat endpoints (require authentication, expect 401)
    info "=== Chat Endpoints (Authentication Required) ==="
    
    ((total_tests++))
    if test_endpoint "/api/chat/health" "401" "Chat Health (Auth Required)"; then
        :
    else
        ((failed_tests++))
    fi
    
    # Admin endpoints (require authentication, expect 401)
    info "=== Admin Endpoints (Authentication Required) ==="
    
    ((total_tests++))
    if test_endpoint "/api/admin/health" "401" "Admin Health (Auth Required)"; then
        :
    else
        ((failed_tests++))
    fi
    
    # Test comprehensive API info endpoint
    info "=== API Information ==="
    
    ((total_tests++))
    if test_endpoint "/" "200" "Root API Information"; then
        :
    else
        ((failed_tests++))
    fi
    
    # Summary
    log "🏁 Test Summary:"
    log "   📊 Total Tests: $total_tests"
    log "   ✅ Passed: $((total_tests - failed_tests))"
    log "   ❌ Failed: $failed_tests"
    
    if [ $failed_tests -eq 0 ]; then
        log "🎉 All tests passed! The comprehensive API is working properly."
        log "📚 API Documentation: $BASE_URL/docs"
        log "🔗 OpenAPI Schema: $BASE_URL/openapi.json"
        log "🏥 Health Check: $BASE_URL/health"
        return 0
    else
        error "❌ $failed_tests tests failed. Please check the deployment."
        return 1
    fi
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [BASE_URL]"
    echo "Example: $0 http://localhost:8000"
    echo "Example: $0 http://104.248.68.3:8000"
    echo ""
    echo "Tests the MIVAA comprehensive API deployment"
    exit 1
fi

# Run tests
main "$@"
