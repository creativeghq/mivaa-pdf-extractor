#!/bin/bash

# Script to update requirements.txt from requirements.in
# This resolves all dependencies locally instead of on the server

set -e

echo "🔧 Updating requirements.txt from requirements.in..."

# Check if we're in the right directory
if [ ! -f "requirements.in" ]; then
    echo "❌ requirements.in not found. Please run this script from the project root."
    exit 1
fi

# Create a temporary virtual environment for dependency resolution
echo "📦 Creating temporary environment for dependency resolution..."
python3 -m venv .temp_venv
source .temp_venv/bin/activate

# Install pip-tools
echo "🛠️ Installing pip-tools..."
pip install --upgrade pip
pip install pip-tools

# Generate locked requirements.txt
echo "🔒 Generating locked requirements.txt..."
pip-compile requirements.in --output-file requirements.txt --resolver=backtracking --verbose

# Clean up temporary environment
deactivate
rm -rf .temp_venv

echo "✅ requirements.txt updated successfully!"
echo "📋 Summary:"
echo "   - All dependencies resolved locally"
echo "   - Exact versions pinned in requirements.txt"
echo "   - Server deployments will be much faster"
echo ""
echo "🚀 Next steps:"
echo "   1. Review the generated requirements.txt"
echo "   2. Commit both requirements.in and requirements.txt"
echo "   3. Deploy - server will use pre-resolved dependencies"
