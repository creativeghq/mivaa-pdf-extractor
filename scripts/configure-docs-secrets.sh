#!/bin/bash

echo "🔧 Configuring MIVAA Documentation via Secrets Management"
echo "========================================================"

echo "⚠️  IMPORTANT: Set these environment variables in your deployment system:"
echo ""
echo "GitHub Secrets (for CI/CD):"
echo "  DOCS_ENABLED=true"
echo "  REDOC_ENABLED=true"
echo ""
echo "Vercel Environment Variables:"
echo "  DOCS_ENABLED=true"
echo "  REDOC_ENABLED=true"
echo ""
echo "Supabase Edge Functions Secrets:"
echo "  DOCS_ENABLED=true"
echo "  REDOC_ENABLED=true"
echo ""
echo "Server Environment (current session only):"
export DOCS_ENABLED=true
export REDOC_ENABLED=true
echo "  ✅ DOCS_ENABLED=true"
echo "  ✅ REDOC_ENABLED=true"
echo ""
echo "🚨 DO NOT CREATE .env FILES - Use proper secret management!"
echo "📚 Documentation: https://v1api.materialshub.gr/docs"
echo "📖 ReDoc: https://v1api.materialshub.gr/redoc"
