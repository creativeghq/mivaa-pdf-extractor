#!/bin/bash

# MIVAA PDF Extractor - Deployment Script with nginx Management
# This script handles deployment with nginx health checking and restart

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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESTART_NGINX="${RESTART_NGINX:-true}"
CHECK_NGINX="${CHECK_NGINX:-true}"

# Help function
show_help() {
    cat << EOF
MIVAA PDF Extractor - Deployment Script with nginx Management

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    --no-nginx              Skip nginx restart
    --check-only            Only check nginx status, don't deploy
    --force-nginx           Force nginx restart even if healthy

ENVIRONMENT VARIABLES:
    RESTART_NGINX=true      Whether to restart nginx (default: true)
    CHECK_NGINX=true        Whether to check nginx status (default: true)

EXAMPLES:
    $0                      # Full deployment with nginx restart
    $0 --no-nginx           # Deploy without nginx restart
    $0 --check-only         # Only check nginx status
    $0 --force-nginx        # Force nginx restart

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-nginx)
            RESTART_NGINX=false
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --force-nginx)
            FORCE_NGINX=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Source deployment functions if available
if [ -f "$PROJECT_ROOT/deploy/setup-server.sh" ]; then
    source "$PROJECT_ROOT/deploy/setup-server.sh"
else
    warn "Deployment functions not found, using basic implementations"
    
    # Basic nginx check function
    check_nginx_status() {
        if systemctl is-active --quiet nginx; then
            log "✅ nginx is running"
            return 0
        else
            error "❌ nginx is not running"
            return 1
        fi
    }
    
    # Basic nginx restart function
    restart_nginx() {
        log "🔄 Restarting nginx..."
        if systemctl restart nginx; then
            log "✅ nginx restarted successfully"
            return 0
        else
            error "❌ Failed to restart nginx"
            return 1
        fi
    }
fi

# Function to check if we're in the right directory
check_environment() {
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi
    
    if [ ! -d "app" ]; then
        error "app directory not found. Please run this script from the project root."
        exit 1
    fi
    
    log "✅ Environment check passed"
}

# Function to deploy application
deploy_application() {
    log "🚀 Starting application deployment..."
    
    # Pull latest code
    log "📥 Pulling latest code..."
    if ! git pull; then
        error "Failed to pull latest code"
        return 1
    fi
    
    # Pull latest Docker images
    log "🐳 Pulling latest Docker images..."
    if ! docker-compose pull; then
        error "Failed to pull Docker images"
        return 1
    fi
    
    # Start/restart containers
    log "🔄 Starting containers..."
    if ! docker-compose up -d; then
        error "Failed to start containers"
        return 1
    fi
    
    # Wait for containers to be ready
    log "⏳ Waiting for containers to be ready..."
    sleep 10
    
    # Check container status
    if docker-compose ps | grep -q "Exit"; then
        error "Some containers failed to start:"
        docker-compose ps
        return 1
    fi
    
    log "✅ Application deployment completed"
    return 0
}

# Function to check application health
check_application_health() {
    log "🏥 Checking application health..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log "✅ Application is healthy"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Application health check failed after $max_attempts attempts"
            return 1
        fi
        
        log "⏳ Waiting for application to be ready (attempt $attempt/$max_attempts)..."
        sleep 5
        ((attempt++))
    done
}

# Main deployment function
main() {
    log "🚀 Starting MIVAA deployment with nginx management..."
    
    # Check environment
    check_environment
    
    # If check-only mode, just check nginx and exit
    if [ "${CHECK_ONLY:-false}" = "true" ]; then
        log "🔍 Check-only mode: checking nginx status..."
        if [ "$CHECK_NGINX" = "true" ]; then
            if command -v check_nginx_for_github &> /dev/null; then
                check_nginx_for_github
            else
                check_nginx_status
            fi
        fi
        exit 0
    fi
    
    # Deploy application
    if ! deploy_application; then
        error "Application deployment failed"
        exit 1
    fi
    
    # Check application health
    if ! check_application_health; then
        error "Application health check failed"
        exit 1
    fi
    
    # nginx management
    if [ "$CHECK_NGINX" = "true" ]; then
        log "🌐 Checking nginx status..."
        
        if [ "${FORCE_NGINX:-false}" = "true" ]; then
            log "🔄 Force nginx restart requested..."
            if command -v restart_nginx &> /dev/null; then
                restart_nginx
            else
                systemctl restart nginx
            fi
        elif [ "$RESTART_NGINX" = "true" ]; then
            # Check if nginx needs restart
            if ! check_nginx_status; then
                log "🔄 nginx needs restart..."
                if command -v restart_nginx &> /dev/null; then
                    restart_nginx
                else
                    systemctl restart nginx
                fi
            else
                log "✅ nginx is healthy, no restart needed"
            fi
        fi
        
        # Final nginx check
        if ! check_nginx_status; then
            error "nginx is not running after deployment"
            exit 1
        fi
        
        # Test if nginx is serving requests
        if curl -f -s http://localhost > /dev/null 2>&1; then
            log "✅ nginx is serving requests"
        else
            warn "nginx is running but not serving requests properly"
        fi
    fi
    
    log "🎉 Deployment completed successfully!"
    log "📋 Summary:"
    log "   ✅ Application deployed and healthy"
    if [ "$CHECK_NGINX" = "true" ]; then
        log "   ✅ nginx checked and healthy"
    fi
    log "   🌐 Service available at: http://localhost"
    log "   📊 Health endpoint: http://localhost:8000/health"
    log "   📚 API docs: http://localhost:8000/docs"
}

# Run main function
main "$@"
