#!/bin/bash

# COMPREHENSIVE PDF PROCESSING END-TO-END TEST RUNNER
# This script runs the comprehensive Python test for PDF processing

set -e  # Exit on error

echo "🧪 Starting Comprehensive PDF Processing End-to-End Test"
echo "======================================================================================================"
echo ""
echo "This test will:"
echo "  1. Test BOTH Claude Vision and GPT Vision models"
echo "  2. Report ALL 12 comprehensive metrics:"
echo "     - Total Products discovered + time taken"
echo "     - Total Pages processed + time taken"
echo "     - Total Chunks created + time taken"
echo "     - Total Images processed + time taken"
echo "     - Total Embeddings created + time taken"
echo "     - Total Errors + time taken"
echo "     - Total Relationships created + time taken"
echo "     - Total Metadata extracted + time taken"
echo "     - Total Memory used + time"
echo "     - Total CPU used + time"
echo "     - Total Cost (AI API usage)"
echo "     - Total Time for entire process"
echo ""
echo "======================================================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the mivaa-pdf-extractor root directory
cd "$SCRIPT_DIR/../.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if httpx is installed
if ! python3 -c "import httpx" 2>/dev/null; then
    echo "❌ Error: httpx module not found. Please install it first."
    echo "   Run: pip install 'httpx<0.25.0'"
    exit 1
fi

# Run the Python test script
echo "🚀 Running comprehensive test..."
echo ""

python3 "$SCRIPT_DIR/comprehensive_pdf_test.py"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================================================"
    echo "✅ Test completed successfully!"
    echo "======================================================================================================"
    exit 0
else
    echo ""
    echo "======================================================================================================"
    echo "❌ Test failed!"
    echo "======================================================================================================"
    exit 1
fi

