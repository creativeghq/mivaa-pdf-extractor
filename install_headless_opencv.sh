#!/bin/bash

# Install OpenCV headless for MIVAA PDF Extractor
echo "🔧 Installing OpenCV headless..."

# Activate virtual environment
source .venv/bin/activate

# Uninstall any existing OpenCV packages to avoid conflicts
echo "🧹 Removing any existing OpenCV packages..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

# Install headless version
echo "📦 Installing opencv-python-headless..."
pip install opencv-python-headless>=4.8.0

# Verify installation
echo "✅ Verifying installation..."
python3 check_opencv.py

echo "🎉 OpenCV headless installation complete!"
