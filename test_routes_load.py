#!/usr/bin/env python3
"""
Simple test to verify saved searches routes load correctly.
Does not require API keys or database connection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_routes_load():
    """Test that the saved searches routes can be imported."""
    print("=" * 80)
    print("Testing Saved Searches Routes Import")
    print("=" * 80)
    
    try:
        from app.api.saved_searches_routes import router
        print(f"\n✓ Routes module imported successfully")
        print(f"  - Number of endpoints: {len(router.routes)}")
        
        # List all endpoints
        print(f"\n✓ Registered endpoints:")
        for route in router.routes:
            methods = ", ".join(route.methods) if hasattr(route, 'methods') else "N/A"
            path = route.path if hasattr(route, 'path') else "N/A"
            print(f"  - {methods:10s} {path}")
        
        print("\n" + "=" * 80)
        print("✓ All routes loaded successfully!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to import routes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_routes_load()
    sys.exit(0 if result else 1)

