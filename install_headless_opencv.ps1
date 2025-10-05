# Install OpenCV headless for MIVAA PDF Extractor
Write-Host "Installing OpenCV headless..." -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "Virtual environment found" -ForegroundColor Green

    # Activate virtual environment
    & ".venv\Scripts\Activate.ps1"

    # Uninstall any existing OpenCV packages to avoid conflicts
    Write-Host "Removing any existing OpenCV packages..." -ForegroundColor Yellow
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless

    # Install headless version
    Write-Host "Installing opencv-python-headless..." -ForegroundColor Green
    pip install opencv-python-headless>=4.8.0

    # Verify installation
    Write-Host "Verifying installation..." -ForegroundColor Green
    python check_opencv.py

    Write-Host "OpenCV headless installation complete!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment not found. Please create one first." -ForegroundColor Red
    Write-Host "Run: python -m venv .venv" -ForegroundColor Yellow
}
