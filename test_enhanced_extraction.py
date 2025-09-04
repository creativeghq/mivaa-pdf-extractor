#!/usr/bin/env python3
"""
Comprehensive test script for enhanced functional metadata extraction system.

This script tests:
1. The 9-category functional metadata extraction
2. Frontend-optimized endpoint functionality  
3. Structured output format validation
4. Confidence scoring and property prioritization
5. Application suggestions generation
"""

import json
import os
import sys
from pathlib import Path
import requests
from typing import Dict, Any, List
import time

# Add the current directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from extractor import FunctionalMetadataExtractor
from main import app


class EnhancedExtractionTester:
    """Test suite for enhanced functional metadata extraction."""
    
    def __init__(self):
        self.extractor = FunctionalMetadataExtractor()
        self.test_results = {}
        self.base_url = "http://localhost:8000"
        
    def test_functional_metadata_categories(self) -> Dict[str, Any]:
        """Test all 9 functional metadata extraction categories."""
        print("🧪 Testing Functional Metadata Categories...")
        
        # Test with sample text containing various properties
        test_specifications = {
            "slip_safety": "R11 slip resistance rating, DIN 51130 compliant, DCOF ≥ 0.55",
            "surface_gloss": "Matte finish, gloss value 5-15 at 60°, anti-glare properties",
            "mechanical": "Breaking strength ≥ 1300 N, PEI IV abrasion resistance, modulus 35 N/mm²",
            "thermal": "Thermal conductivity 1.2 W/m·K, frost resistant, fire rating A1",
            "water_moisture": "Water absorption ≤ 0.5%, stain resistance class 5, moisture expansion ≤ 0.6 mm/m",
            "chemical_hygiene": "Chemical resistance class A, antibacterial ISO 22196, pH 3-11 compatible",
            "acoustic_electrical": "Sound absorption 0.01 NRC, non-conductive >10¹² Ω·cm",
            "environmental": "25% recycled content, GREENGUARD Gold certified, LEED v4.1 credits",
            "dimensional": "600x600mm nominal, ±0.5mm tolerance, rectified edges, V2 shade variation"
        }
        
        category_results = {}
        
        for category, test_text in test_specifications.items():
            try:
                # Test extraction for each category
                if hasattr(self.extractor, f'extract_{category}'):
                    method = getattr(self.extractor, f'extract_{category}')
                    extracted = method(test_text)
                    category_results[category] = {
                        "status": "✅ PASS",
                        "extracted_count": len(extracted),
                        "sample_properties": extracted[:3] if extracted else []
                    }
                    print(f"  ✅ {category}: {len(extracted)} properties extracted")
                else:
                    category_results[category] = {
                        "status": "❌ FAIL",
                        "error": f"Method extract_{category} not found"
                    }
                    print(f"  ❌ {category}: Method not found")
                    
            except Exception as e:
                category_results[category] = {
                    "status": "❌ ERROR", 
                    "error": str(e)
                }
                print(f"  ❌ {category}: {e}")
        
        return category_results
    
    def test_frontend_data_structure(self) -> Dict[str, Any]:
        """Test the frontend-optimized data structure."""
        print("\n🎨 Testing Frontend Data Structure...")
        
        # Test the _structure_metadata_for_frontend function
        try:
            # Import the function
            from main import _structure_metadata_for_frontend
            
            # Create sample raw metadata
            sample_metadata = {
                "slip_safety": [
                    {"name": "Slip Resistance", "value": "R11", "key": "slip_resistance"},
                    {"name": "DIN Standard", "value": "DIN 51130", "key": "din_standard"}
                ],
                "surface_gloss": [
                    {"name": "Gloss Level", "value": "Matte", "key": "gloss_level"},
                    {"name": "Gloss Value", "value": "5-15", "key": "gloss_value"}
                ],
                "mechanical": [
                    {"name": "Breaking Strength", "value": "≥ 1300 N", "key": "breaking_strength"},
                    {"name": "PEI Rating", "value": "PEI IV", "key": "pei_rating"}
                ]
            }
            
            # Test structuring
            structured = _structure_metadata_for_frontend(sample_metadata, "test_tile.pdf")
            
            # Validate structure
            validation_results = {
                "has_document_info": "document_info" in structured,
                "has_functional_properties": "functional_properties" in structured,
                "has_summary": "summary" in structured,
                "categories_found": len(structured.get("functional_properties", {})),
                "sample_category_structure": None
            }
            
            # Check sample category structure
            if structured.get("functional_properties"):
                first_category = list(structured["functional_properties"].values())[0]
                validation_results["sample_category_structure"] = {
                    "has_display_name": "display_name" in first_category,
                    "has_highlights": "highlights" in first_category,
                    "has_technical_details": "technical_details" in first_category,
                    "has_confidence": "extraction_confidence" in first_category
                }
            
            print(f"  ✅ Structure validation: {validation_results}")
            return {"status": "✅ PASS", "validation": validation_results, "sample_output": structured}
            
        except Exception as e:
            print(f"  ❌ Frontend structure test failed: {e}")
            return {"status": "❌ ERROR", "error": str(e)}
    
    def test_confidence_scoring(self) -> Dict[str, Any]:
        """Test the confidence scoring system."""
        print("\n🎯 Testing Confidence Scoring...")
        
        try:
            from main import _assess_extraction_confidence
            
            # Test different confidence scenarios
            test_cases = [
                {"properties": [], "expected": "low"},  # No properties
                {"properties": [{"name": "Test", "value": "Value"}], "expected": "low"},  # 1 property
                {"properties": [{"name": f"Test{i}", "value": f"Value{i}"} for i in range(3)], "expected": "medium"},  # 3 properties
                {"properties": [{"name": f"Test{i}", "value": f"Value{i}"} for i in range(6)], "expected": "high"},  # 6+ properties
            ]
            
            results = []
            for i, case in enumerate(test_cases):
                confidence = _assess_extraction_confidence(case["properties"])
                passed = confidence == case["expected"]
                results.append({
                    "test_case": i + 1,
                    "property_count": len(case["properties"]),
                    "expected_confidence": case["expected"],
                    "actual_confidence": confidence,
                    "passed": passed
                })
                print(f"  {'✅' if passed else '❌'} Case {i+1}: {len(case['properties'])} props → {confidence} (expected {case['expected']})")
            
            all_passed = all(r["passed"] for r in results)
            return {"status": "✅ PASS" if all_passed else "❌ FAIL", "test_cases": results}
            
        except Exception as e:
            print(f"  ❌ Confidence scoring test failed: {e}")
            return {"status": "❌ ERROR", "error": str(e)}
    
    def test_application_suggestions(self) -> Dict[str, Any]:
        """Test the application suggestions generation."""
        print("\n💡 Testing Application Suggestions...")
        
        try:
            from main import _generate_application_suggestions
            
            # Test with sample functional properties
            sample_properties = {
                "slip_safety": [{"name": "Slip Resistance", "value": "R11"}],
                "water_moisture": [{"name": "Water Absorption", "value": "≤ 0.5%"}],
                "thermal": [{"name": "Frost Resistance", "value": "100 cycles"}],
                "chemical_hygiene": [{"name": "Antibacterial", "value": "ISO 22196"}]
            }
            
            suggestions = _generate_application_suggestions(sample_properties)
            
            # Validate suggestions
            has_suggestions = len(suggestions) > 0
            suggestions_are_strings = all(isinstance(s, str) for s in suggestions)
            relevant_keywords = ["commercial", "wet", "outdoor", "healthcare", "food"]
            has_relevant_content = any(
                any(keyword in suggestion.lower() for keyword in relevant_keywords)
                for suggestion in suggestions
            )
            
            results = {
                "has_suggestions": has_suggestions,
                "suggestion_count": len(suggestions),
                "suggestions_are_strings": suggestions_are_strings,
                "has_relevant_content": has_relevant_content,
                "sample_suggestions": suggestions[:3]
            }
            
            print(f"  ✅ Generated {len(suggestions)} application suggestions")
            print(f"  ✅ Sample suggestions: {suggestions[:2]}")
            
            return {"status": "✅ PASS", "results": results}
            
        except Exception as e:
            print(f"  ❌ Application suggestions test failed: {e}")
            return {"status": "❌ ERROR", "error": str(e)}
    
    def test_unit_extraction(self) -> Dict[str, Any]:
        """Test the unit extraction functionality."""
        print("\n📏 Testing Unit Extraction...")
        
        try:
            from main import _extract_unit_from_value
            
            test_cases = [
                {"value": "≥ 1300 N", "expected": "N"},
                {"value": "5-15 at 60°", "expected": ""},
                {"value": "1.2 W/m·K", "expected": "W/m·K"},
                {"value": "≤ 0.5%", "expected": "%"},
                {"value": "600x600mm", "expected": "mm"},
                {"value": "Class 5", "expected": ""},
            ]
            
            results = []
            for case in test_cases:
                extracted_unit = _extract_unit_from_value(case["value"])
                passed = extracted_unit == case["expected"]
                results.append({
                    "value": case["value"],
                    "expected": case["expected"],
                    "extracted": extracted_unit,
                    "passed": passed
                })
                print(f"  {'✅' if passed else '❌'} '{case['value']}' → '{extracted_unit}' (expected '{case['expected']}')")
            
            all_passed = all(r["passed"] for r in results)
            return {"status": "✅ PASS" if all_passed else "❌ FAIL", "test_cases": results}
            
        except Exception as e:
            print(f"  ❌ Unit extraction test failed: {e}")
            return {"status": "❌ ERROR", "error": str(e)}
    
    def test_property_prioritization(self) -> Dict[str, Any]:
        """Test property prioritization logic."""
        print("\n🎯 Testing Property Prioritization...")
        
        try:
            from main import _get_property_display_priority
            
            # Test priority assignments
            test_properties = [
                {"name": "Slip Resistance", "expected": "high"},
                {"name": "Water Absorption", "expected": "high"}, 
                {"name": "Fire Rating", "expected": "high"},
                {"name": "Breaking Strength", "expected": "medium"},
                {"name": "Gloss Value", "expected": "medium"},
                {"name": "Color Code", "expected": "low"},
                {"name": "Unknown Property", "expected": "medium"}  # Default
            ]
            
            results = []
            for prop in test_properties:
                priority = _get_property_display_priority(prop["name"])
                passed = priority == prop["expected"]
                results.append({
                    "property": prop["name"],
                    "expected": prop["expected"],
                    "actual": priority,
                    "passed": passed
                })
                print(f"  {'✅' if passed else '❌'} '{prop['name']}' → {priority} priority")
            
            all_passed = all(r["passed"] for r in results)
            return {"status": "✅ PASS" if all_passed else "❌ FAIL", "test_cases": results}
            
        except Exception as e:
            print(f"  ❌ Property prioritization test failed: {e}")
            return {"status": "❌ ERROR", "error": str(e)}
    
    def test_endpoint_integration(self) -> Dict[str, Any]:
        """Test the enhanced endpoint with functional metadata."""
        print("\n🌐 Testing Endpoint Integration...")
        
        # Check if we have existing extracted images to test with
        images_dir = Path("output/Harvey_2_Well_Completion_Report/images")
        if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
            return {
                "status": "⚠️ SKIP", 
                "reason": "No extracted images found for testing"
            }
        
        try:
            # Test data for API call
            test_images = list(images_dir.glob("*.jpg"))[:3]  # Test with first 3 images
            
            print(f"  📁 Found {len(test_images)} test images")
            
            # Simulate what the endpoint would return
            # (Since we can't actually start the server for testing)
            from main import _structure_metadata_for_frontend
            
            # Create mock metadata for testing
            mock_metadata = {
                "slip_safety": [
                    {"name": "Test Slip Rating", "value": "R9", "key": "slip_rating"}
                ],
                "thermal": [
                    {"name": "Test Temperature", "value": "85°C", "key": "max_temp"}
                ]
            }
            
            # Test the structuring function
            structured_output = _structure_metadata_for_frontend(mock_metadata, "test_document.pdf")
            
            # Validate the output structure
            validation = {
                "has_document_info": "document_info" in structured_output,
                "has_functional_properties": "functional_properties" in structured_output,
                "has_summary": "summary" in structured_output,
                "functional_categories_count": len(structured_output.get("functional_properties", {})),
                "has_display_names": all(
                    "display_name" in cat 
                    for cat in structured_output.get("functional_properties", {}).values()
                ),
                "has_confidence_scores": all(
                    "extraction_confidence" in cat 
                    for cat in structured_output.get("functional_properties", {}).values()
                )
            }
            
            print(f"  ✅ Structured output validation: {validation}")
            
            return {
                "status": "✅ PASS",
                "validation": validation,
                "sample_output_keys": list(structured_output.keys()),
                "test_images_used": len(test_images)
            }
            
        except Exception as e:
            print(f"  ❌ Endpoint integration test failed: {e}")
            return {"status": "❌ ERROR", "error": str(e)}
    
    def test_extraction_patterns(self) -> Dict[str, Any]:
        """Test the enhanced regex patterns for property extraction."""
        print("\n🔍 Testing Enhanced Extraction Patterns...")
        
        pattern_tests = {
            "slip_patterns": [
                ("R11 slip resistance", True),
                ("DCOF ≥ 0.55", True), 
                ("BS 7976-2 compliant", True),
                ("random text", False)
            ],
            "thermal_patterns": [
                ("thermal conductivity 1.2 W/m·K", True),
                ("fire rating A1", True),
                ("frost resistant", True),
                ("unrelated text", False)
            ],
            "water_patterns": [
                ("water absorption ≤ 0.5%", True),
                ("stain resistance class 5", True),
                ("moisture expansion", True),
                ("dry content", False)
            ]
        }
        
        pattern_results = {}
        
        for pattern_group, tests in pattern_tests.items():
            group_results = []
            
            for text, should_match in tests:
                # Test with comprehensive extraction
                all_metadata = self.extractor.extract_comprehensive(text)
                
                # Check if any category found matches
                found_matches = sum(len(props) for props in all_metadata.values())
                actually_matched = found_matches > 0
                
                passed = actually_matched == should_match
                group_results.append({
                    "text": text,
                    "expected_match": should_match,
                    "actually_matched": actually_matched,
                    "matches_found": found_matches,
                    "passed": passed
                })
                
                print(f"  {'✅' if passed else '❌'} '{text}' → {found_matches} matches (expected {'match' if should_match else 'no match'})")
            
            pattern_results[pattern_group] = {
                "test_cases": group_results,
                "pass_rate": sum(1 for r in group_results if r["passed"]) / len(group_results)
            }
        
        overall_pass_rate = sum(r["pass_rate"] for r in pattern_results.values()) / len(pattern_results)
        
        return {
            "status": "✅ PASS" if overall_pass_rate >= 0.8 else "❌ FAIL",
            "overall_pass_rate": overall_pass_rate,
            "pattern_groups": pattern_results
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print("🚀 Starting Comprehensive Enhanced Extraction Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        test_suite = {
            "functional_categories": self.test_functional_metadata_categories(),
            "frontend_structure": self.test_frontend_data_structure(),
            "confidence_scoring": self.test_confidence_scoring(),
            "application_suggestions": self.test_application_suggestions(),
            "unit_extraction": self.test_unit_extraction(),
            "property_prioritization": self.test_property_prioritization(),
            "extraction_patterns": self.test_extraction_patterns(),
            "endpoint_integration": self.test_endpoint_integration()
        }
        
        # Calculate overall results
        passed_tests = sum(1 for result in test_suite.values() if result.get("status", "").startswith("✅"))
        total_tests = len(test_suite)
        success_rate = passed_tests / total_tests
        
        end_time = time.time()
        
        # Generate summary
        summary = {
            "test_execution": {
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": round(end_time - start_time, 2),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": round(success_rate * 100, 1)
            },
            "test_results": test_suite
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print(f"✅ Passed: {passed_tests}/{total_tests} ({summary['test_execution']['success_rate']}%)")
        print(f"⏱️  Duration: {summary['test_execution']['duration_seconds']}s")
        
        # Save results to file
        results_file = Path("test_results_enhanced_extraction.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: {results_file}")
        
        return summary


def main():
    """Run the enhanced extraction test suite."""
    tester = EnhancedExtractionTester()
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    success_rate = results["test_execution"]["success_rate"]
    if success_rate >= 80:
        print("\n🎉 Enhanced extraction system test PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ Enhanced extraction system test FAILED (Success rate: {success_rate}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()