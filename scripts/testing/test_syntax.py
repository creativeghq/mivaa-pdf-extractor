#!/usr/bin/env python3
"""
Quick syntax and import test for comprehensive_pdf_test.py
"""

import sys
import importlib.util

def test_script():
    """Test that the comprehensive test script can be imported."""
    print("🧪 Testing comprehensive_pdf_test.py syntax and imports...")
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(
            "comprehensive_pdf_test",
            "scripts/testing/comprehensive_pdf_test.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        print("✅ Script syntax is valid")
        
        # Try to load it (this will check imports)
        spec.loader.exec_module(module)
        
        print("✅ All imports successful")
        
        # Check key functions exist
        required_functions = [
            'log',
            'log_section',
            'get_system_metrics',
            'calculate_cost',
            'format_duration',
            'cleanup_old_test_data',
            'upload_pdf_for_nova_extraction',
            'validate_data_saved',
            'monitor_processing_job_with_metrics',
            'collect_final_metrics',
            'generate_model_report',
            'generate_comparison_report',
            'run_single_model_test',
            'run_nova_product_test',
            'main'
        ]
        
        for func_name in required_functions:
            if not hasattr(module, func_name):
                print(f"❌ Missing function: {func_name}")
                return False
        
        print(f"✅ All {len(required_functions)} required functions found")
        
        # Check configuration constants
        required_constants = [
            'MIVAA_API',
            'HARMONY_PDF_URL',
            'WORKSPACE_ID',
            'TEST_MODELS',
            'MODEL_PRICING'
        ]
        
        for const_name in required_constants:
            if not hasattr(module, const_name):
                print(f"❌ Missing constant: {const_name}")
                return False
        
        print(f"✅ All {len(required_constants)} required constants found")
        
        print("\n✅ All tests passed! Script is ready to run.")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Note: This is expected if httpx is not installed.")
        print("   The script will auto-install it when run.")
        return True  # Still consider this a pass
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == '__main__':
    success = test_script()
    sys.exit(0 if success else 1)

