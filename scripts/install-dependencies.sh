#!/bin/bash

# MIVAA PDF Extractor - Dependency Installation Script
# Based on deploy.bak workflow process
# This script ensures all dependencies are installed correctly

set -e  # Exit on error

echo "🚀 MIVAA Dependency Installation"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="$(dirname "$SCRIPT_DIR")"

cd "$APP_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    log_error "Virtual environment not found. Please create it first:"
    echo "  python3.9 -m venv .venv"
    exit 1
fi

# Activate virtual environment
log_step "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
log_step "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install critical dependencies first (to avoid conflicts)
log_step "Installing critical dependencies..."

# Install exact versions to prevent conflicts
log_step "Installing pandas with exact version..."
pip install pandas==2.1.4 --no-deps

log_step "Installing pytz (pandas dependency)..."
pip install pytz

log_step "Installing numpy..."
pip install numpy==1.26.4

log_step "Installing pillow..."
pip install 'Pillow>=10.2.0,<11.0.0'

log_step "Installing imageio..."
pip install 'imageio>=2.31.0,<2.32.0'

# Install httpx with correct version
log_step "Installing httpx..."
pip install 'httpx>=0.24.0,<0.25.0'

# Install sentry-sdk
log_step "Installing sentry-sdk..."
pip install 'sentry-sdk[fastapi]>=2.35.0'

# Install all missing dependencies from requirements.txt errors
log_step "Installing missing dependencies..."
pip install deprecation distro exceptiongroup httpcore dataclasses-json \
    dirtyjson nest-asyncio sqlalchemy tiktoken typing-inspect wrapt \
    ftfy beautifulsoup4 bs4 jiter strenum websockets

# Install striprtf with correct version
log_step "Installing striprtf..."
pip install 'striprtf>=0.0.26,<0.0.27'

# Install websockets with correct version
log_step "Installing websockets..."
pip install 'websockets>=11,<13'

# Now install remaining requirements
log_step "Installing remaining requirements from requirements.txt..."
pip install -r requirements.txt --no-deps || true

# Install any missing dependencies
log_step "Installing any missing dependencies..."
pip install -r requirements.txt || true

log_success "Dependency installation complete!"
echo ""
echo "📋 Verification:"
python -c "
import sys
packages = [
    'fastapi', 'uvicorn', 'pydantic', 'supabase', 'httpx', 
    'requests', 'pandas', 'numpy', 'sentry_sdk', 'deprecation'
]
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError as e:
        print(f'  ❌ {pkg}: {e}')
        failed.append(pkg)

if failed:
    print(f'\n❌ Failed to import: {', '.join(failed)}')
    sys.exit(1)
else:
    print('\n✅ All critical packages imported successfully!')
"

echo ""
log_success "Installation complete! You can now restart the service."
echo "  sudo systemctl restart mivaa-pdf-extractor"

