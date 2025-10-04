#!/bin/bash

# MIVAA PDF Extractor - Migration Script from Legacy Service to Docker
# This script migrates from the legacy systemd service to Docker deployment

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
APP_DIR="/opt/mivaa-pdf-extractor"
LEGACY_SERVICE="mivaa-pdf-extractor"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root for service management"
   exit 1
fi

log "🚀 Starting migration from legacy service to Docker deployment..."

# Step 1: Check current service status
check_legacy_service() {
    log "🔍 Checking legacy service status..."
    
    if systemctl list-units --full -all | grep -Fq "$LEGACY_SERVICE.service"; then
        log "📋 Legacy service found: $LEGACY_SERVICE"
        
        if systemctl is-active --quiet "$LEGACY_SERVICE"; then
            log "⚠️  Legacy service is currently running"
            return 0
        else
            log "⏹️  Legacy service exists but is not running"
            return 1
        fi
    else
        log "✅ No legacy service found - clean Docker deployment"
        return 2
    fi
}

# Step 2: Stop and disable legacy service
stop_legacy_service() {
    log "⏹️  Stopping legacy service..."
    
    if systemctl is-active --quiet "$LEGACY_SERVICE"; then
        systemctl stop "$LEGACY_SERVICE"
        log "✅ Legacy service stopped"
    fi
    
    if systemctl is-enabled --quiet "$LEGACY_SERVICE" 2>/dev/null; then
        systemctl disable "$LEGACY_SERVICE"
        log "✅ Legacy service disabled"
    fi
    
    # Remove service file if it exists
    if [ -f "/etc/systemd/system/$LEGACY_SERVICE.service" ]; then
        rm -f "/etc/systemd/system/$LEGACY_SERVICE.service"
        systemctl daemon-reload
        log "✅ Legacy service file removed"
    fi
}

# Step 3: Ensure Docker is ready
check_docker() {
    log "🐳 Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please run server setup first."
        exit 1
    fi
    
    if ! systemctl is-active --quiet docker; then
        log "🔄 Starting Docker service..."
        systemctl start docker
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please run server setup first."
        exit 1
    fi
    
    log "✅ Docker is ready"
}

# Step 4: Prepare Docker environment
prepare_docker_environment() {
    log "📁 Preparing Docker environment..."
    
    # Ensure we're in the right directory
    if [ ! -d "$APP_DIR" ]; then
        error "Application directory not found: $APP_DIR"
        exit 1
    fi
    
    cd "$APP_DIR"
    
    # Check for required files
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found in $APP_DIR"
        exit 1
    fi
    
    if [ ! -f "Dockerfile" ]; then
        error "Dockerfile not found in $APP_DIR"
        exit 1
    fi
    
    # Create required directories
    mkdir -p logs uploads temp
    
    # Set proper permissions
    chown -R 1000:1000 logs uploads temp
    
    log "✅ Docker environment prepared"
}

# Step 5: Start Docker services
start_docker_services() {
    log "🚀 Starting Docker services..."
    
    cd "$APP_DIR"
    
    # Pull latest images
    log "📥 Pulling latest Docker images..."
    docker-compose pull
    
    # Start services
    log "🔄 Starting containers..."
    docker-compose up -d
    
    # Wait for services to be ready
    log "⏳ Waiting for services to be ready..."
    sleep 15
    
    # Check container status
    if docker-compose ps | grep -q "Exit"; then
        error "Some containers failed to start:"
        docker-compose ps
        return 1
    fi
    
    log "✅ Docker services started successfully"
}

# Step 6: Verify deployment
verify_deployment() {
    log "🔍 Verifying deployment..."
    
    # Check container health
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log "✅ Application health check passed"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Application health check failed after $max_attempts attempts"
            return 1
        fi
        
        log "⏳ Waiting for application to be ready (attempt $attempt/$max_attempts)..."
        sleep 5
        ((attempt++))
    done
    
    # Test API endpoints
    log "🧪 Testing API endpoints..."
    
    # Test legacy redirect
    if curl -f -s http://localhost:8000/ > /dev/null; then
        log "✅ Legacy API redirect is working"
    else
        warn "Legacy API redirect test failed"
    fi
    
    # Test comprehensive API (if available)
    if curl -f -s http://localhost:8000/docs > /dev/null; then
        log "✅ API documentation is accessible"
    else
        warn "API documentation test failed"
    fi
    
    log "✅ Deployment verification completed"
}

# Step 7: Update nginx configuration (if needed)
update_nginx_config() {
    log "🌐 Checking nginx configuration..."
    
    # Check if nginx is installed and running
    if command -v nginx &> /dev/null; then
        if systemctl is-active --quiet nginx; then
            log "✅ nginx is running"
            
            # Test nginx configuration
            if nginx -t 2>/dev/null; then
                log "✅ nginx configuration is valid"
                
                # Reload nginx to ensure it's using the latest config
                systemctl reload nginx
                log "✅ nginx configuration reloaded"
            else
                warn "nginx configuration has issues - please check manually"
            fi
        else
            warn "nginx is installed but not running"
        fi
    else
        info "nginx is not installed - skipping nginx configuration"
    fi
}

# Main migration function
main() {
    log "🚀 Starting MIVAA migration to Docker deployment..."
    
    # Check legacy service status
    check_legacy_service
    local legacy_status=$?
    
    if [ $legacy_status -eq 0 ]; then
        warn "Legacy service is running - will stop and migrate to Docker"
        stop_legacy_service
    elif [ $legacy_status -eq 1 ]; then
        info "Legacy service exists but not running - will clean up and migrate"
        stop_legacy_service
    else
        info "No legacy service found - proceeding with Docker deployment"
    fi
    
    # Prepare Docker environment
    check_docker
    prepare_docker_environment
    
    # Start Docker services
    if ! start_docker_services; then
        error "Failed to start Docker services"
        exit 1
    fi
    
    # Verify deployment
    if ! verify_deployment; then
        error "Deployment verification failed"
        exit 1
    fi
    
    # Update nginx if needed
    update_nginx_config
    
    log "🎉 Migration completed successfully!"
    log "📋 Summary:"
    log "   ✅ Legacy service stopped and disabled"
    log "   ✅ Docker services started"
    log "   ✅ Application is healthy and responding"
    log "   🌐 Service available at: http://localhost:8000"
    log "   📚 API docs: http://localhost:8000/docs"
    log "   📊 Health endpoint: http://localhost:8000/health"
    
    info "🔧 Management commands:"
    info "   • View logs: docker-compose logs -f"
    info "   • Check status: docker-compose ps"
    info "   • Restart: docker-compose restart"
    info "   • Stop: docker-compose down"
    info "   • Update: git pull && docker-compose pull && docker-compose up -d"
}

# Run main function
main "$@"
