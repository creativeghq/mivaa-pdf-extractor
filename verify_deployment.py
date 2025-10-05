#!/usr/bin/env python3
"""
Deployment verification script for MIVAA PDF Extractor
Checks if all required packages are properly installed
"""

import sys
import importlib
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Critical packages that must be available
CRITICAL_PACKAGES = {
    'fastapi': 'FastAPI web framework',
    'uvicorn': 'ASGI server',
    'pydantic': 'Data validation',
    'supabase': 'Database client',
    'pymupdf4llm': 'PDF processing',
    'fitz': 'PyMuPDF (PDF manipulation)',
    'numpy': 'Numerical computing',
    'pandas': 'Data manipulation',
    'requests': 'HTTP client',
    'httpx': 'Async HTTP client',
    'aiofiles': 'Async file operations',
    'PIL': 'Image processing (Pillow)',
    'cv2': 'OpenCV (headless)',
}

# Optional packages that enhance functionality
OPTIONAL_PACKAGES = {
    'llama_index': 'RAG framework',
    'transformers': 'ML transformers',
    'scikit-learn': 'Machine learning',
    'imageio': 'Image I/O',
    'easyocr': 'OCR engine',
    'nltk': 'Natural language processing',
    'matplotlib': 'Plotting library',
}

def check_package(package_name: str, description: str) -> Tuple[bool, str]:
    """Check if a package is available and get version info"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def verify_packages() -> Dict[str, Dict]:
    """Verify all packages and return status report"""
    results = {
        'critical': {},
        'optional': {},
        'summary': {
            'critical_missing': 0,
            'optional_missing': 0,
            'total_critical': len(CRITICAL_PACKAGES),
            'total_optional': len(OPTIONAL_PACKAGES)
        }
    }
    
    print("🔍 Verifying MIVAA PDF Extractor Dependencies")
    print("=" * 50)
    
    # Check critical packages
    print("\n📦 Critical Packages:")
    for package, description in CRITICAL_PACKAGES.items():
        available, version_or_error = check_package(package, description)
        results['critical'][package] = {
            'available': available,
            'version': version_or_error if available else None,
            'error': version_or_error if not available else None,
            'description': description
        }
        
        if available:
            print(f"  ✅ {package:<15} v{version_or_error:<10} - {description}")
        else:
            print(f"  ❌ {package:<15} MISSING      - {description}")
            results['summary']['critical_missing'] += 1
    
    # Check optional packages
    print("\n🔧 Optional Packages:")
    for package, description in OPTIONAL_PACKAGES.items():
        available, version_or_error = check_package(package, description)
        results['optional'][package] = {
            'available': available,
            'version': version_or_error if available else None,
            'error': version_or_error if not available else None,
            'description': description
        }
        
        if available:
            print(f"  ✅ {package:<15} v{version_or_error:<10} - {description}")
        else:
            print(f"  ⚠️  {package:<15} MISSING      - {description}")
            results['summary']['optional_missing'] += 1
    
    return results

def test_application_startup():
    """Test if the application can start successfully"""
    print("\n🚀 Testing Application Startup:")
    
    try:
        from app.config import get_settings
        settings = get_settings()
        print(f"  ✅ Configuration loaded - {settings.app_name}")
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False
    
    try:
        from app.main import create_app
        app = create_app()
        print(f"  ✅ FastAPI application created")
    except Exception as e:
        print(f"  ❌ Application creation failed: {e}")
        return False
    
    return True

def main():
    """Main verification function"""
    results = verify_packages()
    startup_success = test_application_startup()
    
    # Print summary
    print("\n📊 Deployment Summary:")
    print("=" * 30)
    
    critical_ok = results['summary']['critical_missing'] == 0
    optional_ok = results['summary']['optional_missing'] == 0
    
    print(f"Critical packages: {results['summary']['total_critical'] - results['summary']['critical_missing']}/{results['summary']['total_critical']} ({'✅ OK' if critical_ok else '❌ MISSING'})")
    print(f"Optional packages: {results['summary']['total_optional'] - results['summary']['optional_missing']}/{results['summary']['total_optional']} ({'✅ OK' if optional_ok else '⚠️ SOME MISSING'})")
    print(f"Application startup: {'✅ OK' if startup_success else '❌ FAILED'}")
    
    # Overall status
    if critical_ok and startup_success:
        print("\n🎉 Deployment verification PASSED!")
        print("   The MIVAA PDF Extractor is ready for production.")
        return 0
    else:
        print("\n🚨 Deployment verification FAILED!")
        if not critical_ok:
            print("   Critical packages are missing.")
        if not startup_success:
            print("   Application startup failed.")
        return 1

def get_package_status_json():
    """Return package status as JSON for API consumption"""
    import json

    results = verify_packages()
    startup_success = test_application_startup()

    return json.dumps({
        'packages': results,
        'startup_success': startup_success,
        'timestamp': datetime.utcnow().isoformat(),
        'deployment_ready': results['summary']['critical_missing'] == 0 and startup_success
    }, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        print(get_package_status_json())
    else:
        sys.exit(main())
