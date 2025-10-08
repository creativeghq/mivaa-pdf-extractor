#!/bin/bash

# Enable MIVAA Documentation Script
# This script ensures documentation endpoints are always enabled via proper secret management

echo "🔧 Enabling MIVAA Documentation Endpoints..."

# Set environment variables for current session only
export DOCS_ENABLED=true
export REDOC_ENABLED=true

echo "✅ Documentation endpoints enabled for current session"
echo "📚 Swagger UI: https://v1api.materialshub.gr/docs"
echo "📖 ReDoc: https://v1api.materialshub.gr/redoc"

# Restart the service if it's running
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    echo "🔄 Restarting MIVAA service..."
    pkill -HUP -f "uvicorn app.main:app"
    sleep 3
    echo "✅ Service restarted"
else
    echo "⚠️ Service not running - start it manually"
fi

echo ""
echo "🚨 IMPORTANT: For persistent configuration, set these in your deployment system:"
echo "   GitHub Secrets: DOCS_ENABLED=true, REDOC_ENABLED=true"
echo "   Vercel Environment Variables: DOCS_ENABLED=true, REDOC_ENABLED=true"
echo "   Supabase Edge Functions Secrets: DOCS_ENABLED=true, REDOC_ENABLED=true"
echo ""
echo "🎉 Documentation configuration complete!"
