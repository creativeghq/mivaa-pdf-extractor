#!/bin/bash
set -e
cd /var/www/mivaa-pdf-extractor
source .venv/bin/activate

echo "🚀 Installing critical dependencies..."

# Kill any running pip processes
pkill -9 -f "pip install" || true

# Upgrade pip
python -m pip install --upgrade pip --quiet

# Install critical missing dependencies
echo "Installing pandas and pytz..."
pip install pandas==2.1.4 pytz --quiet

echo "Installing sentry-sdk..."
pip install 'sentry-sdk[fastapi]>=2.35.0' --quiet

echo "Installing missing core dependencies..."
pip install deprecation distro exceptiongroup httpcore dataclasses-json \
    dirtyjson nest-asyncio sqlalchemy tiktoken typing-inspect wrapt \
    ftfy beautifulsoup4 bs4 jiter strenum --quiet

echo "Installing striprtf and websockets with correct versions..."
pip install 'striprtf>=0.0.26,<0.0.27' 'websockets>=11,<13' --quiet

echo "✅ Critical dependencies installed!"
echo "Verifying imports..."
python -c "import sentry_sdk, deprecation, pandas, pytz; print('✅ All critical packages work!')"
