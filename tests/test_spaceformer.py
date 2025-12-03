"""
Test script for Spaceformer Spatial Analysis
Tests the complete workflow from API call to database storage
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.spaceformer_service import SpaceformerService
from app.core.config import get_settings

# Test image URL (sample room image)
TEST_IMAGE_URL = "https://images.unsplash.com/photo-1616486338812-3dadae4b4ace?w=800"

async def test_spaceformer_full_analysis():
    """Test full spatial analysis"""
    print("\n" + "="*80)
    print("TEST 1: Full Spatial Analysis")
    print("="*80)
    
    service = SpaceformerService()
    
    try:
        result = await service.analyze_space(
            image_url=TEST_IMAGE_URL,
            room_type="living_room",
            room_dimensions={"width": 5.0, "height": 3.0, "depth": 4.0},
            user_preferences={
                "style": "modern",
                "color_preferences": ["white", "beige"],
                "material_preferences": ["wood", "ceramic"]
            },
            analysis_type="full",
            user_id="test_user_123",
            workspace_id="test_workspace_456"
        )
        
        print("\n✅ Analysis completed successfully!")
        print(f"   Analysis ID: {result['analysis_id']}")
        print(f"   Confidence Score: {result['confidence_score']:.2f}")
        print(f"   Processing Time: {result['processing_time_ms']}ms")
        print(f"   Spatial Features: {len(result['spatial_features'])}")
        print(f"   Layout Suggestions: {len(result['layout_suggestions'])}")
        print(f"   Material Placements: {len(result['material_placements'])}")
        
        if result.get('accessibility_analysis'):
            print(f"   Accessibility Score: {result['accessibility_analysis'].get('compliance_score', 0):.2f}")
            print(f"   ADA Compliance: {result['accessibility_analysis'].get('ada_compliance', False)}")
        
        if result.get('flow_optimization'):
            print(f"   Flow Efficiency: {result['flow_optimization'].get('efficiency_score', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_spaceformer_layout_only():
    """Test layout-only analysis"""
    print("\n" + "="*80)
    print("TEST 2: Layout-Only Analysis")
    print("="*80)
    
    service = SpaceformerService()
    
    try:
        result = await service.analyze_space(
            image_url=TEST_IMAGE_URL,
            room_type="bedroom",
            analysis_type="layout",
            user_id="test_user_123",
            workspace_id="test_workspace_456"
        )
        
        print("\n✅ Layout analysis completed!")
        print(f"   Analysis ID: {result['analysis_id']}")
        print(f"   Layout Suggestions: {len(result['layout_suggestions'])}")
        print(f"   Processing Time: {result['processing_time_ms']}ms")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_spaceformer_accessibility():
    """Test accessibility analysis"""
    print("\n" + "="*80)
    print("TEST 3: Accessibility Analysis")
    print("="*80)
    
    service = SpaceformerService()
    
    try:
        result = await service.analyze_space(
            image_url=TEST_IMAGE_URL,
            room_type="bathroom",
            analysis_type="accessibility",
            user_id="test_user_123",
            workspace_id="test_workspace_456"
        )
        
        print("\n✅ Accessibility analysis completed!")
        print(f"   Analysis ID: {result['analysis_id']}")
        print(f"   Compliance Score: {result['accessibility_analysis'].get('compliance_score', 0):.2f}")
        print(f"   ADA Compliance: {result['accessibility_analysis'].get('ada_compliance', False)}")
        print(f"   Recommendations: {len(result['accessibility_analysis'].get('recommendations', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SPACEFORMER SPATIAL ANALYSIS - END-TO-END TESTS")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(await test_spaceformer_full_analysis())
    results.append(await test_spaceformer_layout_only())
    results.append(await test_spaceformer_accessibility())
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

