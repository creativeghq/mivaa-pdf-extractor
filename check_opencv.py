#!/usr/bin/env python3
"""
Quick check for OpenCV headless installation
"""

import sys

def check_opencv():
    """Check if OpenCV headless is properly installed"""
    
    print("🔍 Checking OpenCV Installation...")
    print("=" * 40)
    
    try:
        import cv2
        print(f"✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
        
        # Check if it's the headless version
        try:
            # This should work in headless version
            import numpy as np
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            print(f"✅ Basic image operations work")
            
            # Check if GUI functions are available (should not be in headless)
            try:
                cv2.imshow('test', test_image)
                print(f"⚠️  GUI functions available (not headless version)")
                cv2.destroyAllWindows()
            except cv2.error:
                print(f"✅ GUI functions disabled (headless version confirmed)")
            
        except Exception as e:
            print(f"❌ Basic operations failed: {e}")
            return False
            
        print(f"\n🎉 OpenCV is properly installed and functional!")
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV not available: {e}")
        print(f"\n💡 To install OpenCV headless:")
        print(f"   pip install opencv-python-headless>=4.8.0")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = check_opencv()
    sys.exit(0 if success else 1)
