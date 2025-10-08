#!/bin/bash

echo "🔍 MIVAA Documentation Verification"
echo "=================================="

# Test local endpoints
echo "📍 Testing local endpoints..."
DOCS_LOCAL=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs)
REDOC_LOCAL=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/redoc)
OPENAPI_LOCAL=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/openapi.json)

echo "   /docs: $DOCS_LOCAL"
echo "   /redoc: $REDOC_LOCAL"
echo "   /openapi.json: $OPENAPI_LOCAL"

# Test external endpoints
echo ""
echo "🌐 Testing external endpoints..."
DOCS_EXT=$(curl -s -o /dev/null -w "%{http_code}" https://v1api.materialshub.gr/docs)
REDOC_EXT=$(curl -s -o /dev/null -w "%{http_code}" https://v1api.materialshub.gr/redoc)
OPENAPI_EXT=$(curl -s -o /dev/null -w "%{http_code}" https://v1api.materialshub.gr/openapi.json)

echo "   https://v1api.materialshub.gr/docs: $DOCS_EXT"
echo "   https://v1api.materialshub.gr/redoc: $REDOC_EXT"
echo "   https://v1api.materialshub.gr/openapi.json: $OPENAPI_EXT"

# Check configuration
echo ""
echo "⚙️ Configuration status..."
cd /var/www/mivaa-pdf-extractor
DOCS_ENABLED=$(python3.9 -c "from app.config import get_settings; print(get_settings().docs_enabled)" 2>/dev/null)
REDOC_ENABLED=$(python3.9 -c "from app.config import get_settings; print(get_settings().redoc_enabled)" 2>/dev/null)

echo "   DOCS_ENABLED: $DOCS_ENABLED"
echo "   REDOC_ENABLED: $REDOC_ENABLED"

# Overall status
echo ""
if [[ "$DOCS_EXT" == "200" && "$REDOC_EXT" == "200" && "$OPENAPI_EXT" == "200" ]]; then
    echo "✅ ALL DOCUMENTATION ENDPOINTS WORKING!"
    echo "📚 Swagger UI: https://v1api.materialshub.gr/docs"
    echo "📖 ReDoc: https://v1api.materialshub.gr/redoc"
    echo "🔧 OpenAPI Spec: https://v1api.materialshub.gr/openapi.json"
else
    echo "❌ Some endpoints are not working properly"
fi

echo ""
echo "🎉 Documentation verification complete!"
