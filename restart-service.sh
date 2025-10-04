#!/bin/bash

# MIVAA Service Restart Script
# This script ensures the service runs with the new comprehensive API (app/main.py)

echo "🔄 MIVAA Service Restart - Switching to Comprehensive API"
echo "=================================================="

# Function to check if service is running
check_service() {
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get service info
get_service_info() {
    curl -s http://localhost:8000/ 2>/dev/null || echo '{"error": "Service not responding"}'
}

echo "📊 Current Service Status:"
if check_service; then
    echo "✅ Service is running"
    echo "📋 Service Info:"
    get_service_info | python3 -m json.tool 2>/dev/null || echo "Could not parse service info"
else
    echo "❌ Service is not running"
fi

echo ""
echo "🛑 Stopping any running services..."

# Stop Docker containers
if command -v docker-compose &> /dev/null; then
    echo "🐳 Stopping Docker Compose services..."
    docker-compose down
fi

# Kill any uvicorn processes
echo "🔪 Killing any uvicorn processes..."
pkill -f uvicorn || echo "No uvicorn processes found"

# Wait a moment
sleep 2

echo ""
echo "🚀 Starting service with new comprehensive API..."

# Check if we're in Docker environment
if [ -f "docker-compose.yml" ]; then
    echo "🐳 Starting with Docker Compose..."
    docker-compose up -d
    
    echo "⏳ Waiting for service to start..."
    sleep 10
    
    # Check if service started successfully
    if check_service; then
        echo "✅ Service started successfully!"
        echo ""
        echo "📋 New Service Info:"
        get_service_info | python3 -m json.tool 2>/dev/null
        echo ""
        echo "🌐 Available endpoints:"
        echo "  - Swagger UI: http://localhost:8000/docs"
        echo "  - ReDoc: http://localhost:8000/redoc"
        echo "  - OpenAPI Schema: http://localhost:8000/openapi.json"
        echo "  - Health Check: http://localhost:8000/health"
        echo "  - Service Info: http://localhost:8000/"
    else
        echo "❌ Service failed to start. Check logs:"
        docker-compose logs
    fi
else
    echo "🐍 Starting with uvicorn directly..."
    echo "Command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
    echo ""
    echo "Run this command manually:"
    echo "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
fi

echo ""
echo "🎯 Key Changes:"
echo "  ✅ Legacy main.py disabled"
echo "  ✅ New app/main.py with 37+ endpoints"
echo "  ✅ JWT authentication enabled"
echo "  ✅ RAG system integration"
echo "  ✅ Vector search capabilities"
echo "  ✅ Performance monitoring"
echo ""
echo "📚 Documentation:"
echo "  - API Documentation: ./API_DOCUMENTATION.md"
echo "  - Deployment Guide: ./deploy/README.md"
echo "  - Service Overview: ./README.md"
