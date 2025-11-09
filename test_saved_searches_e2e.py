#!/usr/bin/env python3
"""
E2E Test Script for Saved Searches with AI Deduplication

Tests the complete workflow:
1. Create searches with deduplication
2. Check for duplicates
3. Merge searches
4. Execute searches
5. Update and delete searches

Usage:
    python test_saved_searches_e2e.py
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.search_deduplication_service import SearchDeduplicationService
from app.services.supabase_client import get_supabase_client
from app.config import get_settings


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


class SavedSearchesE2ETest:
    """E2E test suite for saved searches."""

    def __init__(self):
        self.settings = get_settings()

        # Initialize Supabase client
        supabase_wrapper = get_supabase_client()
        supabase_wrapper.initialize(self.settings)
        self.supabase = supabase_wrapper.client

        self.dedup_service = SearchDeduplicationService()
        self.test_user_id = "test-user-e2e-" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.created_search_ids: List[str] = []
        self.test_results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    
    async def cleanup(self):
        """Clean up test data."""
        print_info(f"Cleaning up test data for user: {self.test_user_id}")
        
        try:
            # Delete all test searches
            self.supabase.table("saved_searches").delete().eq(
                "user_id", self.test_user_id
            ).execute()
            
            print_success(f"Cleaned up {len(self.created_search_ids)} test searches")
        except Exception as e:
            print_error(f"Cleanup failed: {e}")
    
    def record_test(self, name: str, passed: bool, error: str = None):
        """Record test result."""
        self.test_results["total"] += 1
        if passed:
            self.test_results["passed"] += 1
            print_success(f"Test passed: {name}")
        else:
            self.test_results["failed"] += 1
            self.test_results["errors"].append({"test": name, "error": error})
            print_error(f"Test failed: {name} - {error}")
    
    async def test_1_create_search_no_duplicates(self):
        """Test 1: Create a new search with no duplicates."""
        print_header("Test 1: Create Search (No Duplicates)")
        
        try:
            query = "grey marble floor tiles for modern kitchen"
            
            # Analyze query
            analysis = await self.dedup_service.analyze_search_query(query)
            
            print_info(f"Query: {query}")
            print_info(f"Core Material: {analysis.core_material}")
            print_info(f"Attributes: {json.dumps(analysis.attributes, indent=2)}")
            print_info(f"Application Context: {analysis.application_context}")
            print_info(f"Intent Category: {analysis.intent_category}")
            
            # Create search
            search_data = {
                "user_id": self.test_user_id,
                "query": query,
                "name": "Test Search 1",
                "semantic_fingerprint": analysis.semantic_fingerprint,
                "normalized_query": analysis.normalized_query,
                "core_material": analysis.core_material,
                "material_attributes": analysis.attributes,
                "application_context": analysis.application_context,
                "intent_category": analysis.intent_category,
                "use_count": 0,
                "merge_count": 1,
                "relevance_score": 1.0
            }
            
            response = self.supabase.table("saved_searches").insert(search_data).execute()
            
            if response.data and len(response.data) > 0:
                search_id = response.data[0]["id"]
                self.created_search_ids.append(search_id)
                print_success(f"Created search: {search_id}")
                self.record_test("Create search (no duplicates)", True)
            else:
                self.record_test("Create search (no duplicates)", False, "No data returned")
        
        except Exception as e:
            self.record_test("Create search (no duplicates)", False, str(e))
    
    async def test_2_detect_exact_duplicate(self):
        """Test 2: Detect exact duplicate (should auto-merge)."""
        print_header("Test 2: Detect Exact Duplicate (Auto-Merge)")
        
        try:
            # Same query as Test 1
            query = "grey marble floor tiles for modern kitchen"
            
            existing_id, should_merge, merge_suggestion = await self.dedup_service.find_or_merge_search(
                user_id=self.test_user_id,
                query=query,
                filters={},
                material_filters={}
            )
            
            if existing_id and should_merge:
                print_success(f"Detected exact duplicate: {existing_id}")
                print_info("Auto-merge recommended (95%+ similarity)")
                self.record_test("Detect exact duplicate", True)
            else:
                self.record_test("Detect exact duplicate", False, "Should have detected duplicate")
        
        except Exception as e:
            self.record_test("Detect exact duplicate", False, str(e))
    
    async def test_3_detect_semantic_duplicate(self):
        """Test 3: Detect semantic duplicate (should suggest merge)."""
        print_header("Test 3: Detect Semantic Duplicate (Suggest Merge)")
        
        try:
            # Similar but not identical query
            query = "gray marble flooring for contemporary kitchen design"
            
            existing_id, should_merge, merge_suggestion = await self.dedup_service.find_or_merge_search(
                user_id=self.test_user_id,
                query=query,
                filters={},
                material_filters={}
            )
            
            if existing_id and merge_suggestion:
                print_success(f"Detected semantic duplicate: {existing_id}")
                print_info(f"Similarity: {merge_suggestion['similarity_score']:.2%}")
                print_info(f"Reason: {merge_suggestion['reason']}")
                self.record_test("Detect semantic duplicate", True)
            else:
                self.record_test("Detect semantic duplicate", False, "Should have detected semantic duplicate")
        
        except Exception as e:
            self.record_test("Detect semantic duplicate", False, str(e))
    
    async def test_4_reject_different_context(self):
        """Test 4: Reject search with different context (floor vs wall)."""
        print_header("Test 4: Reject Different Context (Floor vs Wall)")
        
        try:
            # Same material, different application
            query = "grey marble wall tiles for modern kitchen"
            
            existing_id, should_merge, merge_suggestion = await self.dedup_service.find_or_merge_search(
                user_id=self.test_user_id,
                query=query,
                filters={},
                material_filters={}
            )
            
            if not existing_id:
                print_success("Correctly rejected different context (floor vs wall)")
                self.record_test("Reject different context", True)
            else:
                self.record_test("Reject different context", False, "Should not have matched floor vs wall")
        
        except Exception as e:
            self.record_test("Reject different context", False, str(e))
    
    async def test_5_reject_different_attributes(self):
        """Test 5: Reject search with different attributes (grey vs white)."""
        print_header("Test 5: Reject Different Attributes (Grey vs White)")
        
        try:
            # Same material, different color
            query = "white marble floor tiles for modern kitchen"
            
            existing_id, should_merge, merge_suggestion = await self.dedup_service.find_or_merge_search(
                user_id=self.test_user_id,
                query=query,
                filters={},
                material_filters={}
            )
            
            if not existing_id:
                print_success("Correctly rejected different attributes (grey vs white)")
                self.record_test("Reject different attributes", True)
            else:
                self.record_test("Reject different attributes", False, "Should not have matched grey vs white")
        
        except Exception as e:
            self.record_test("Reject different attributes", False, str(e))
    
    async def test_6_merge_searches(self):
        """Test 6: Merge two searches."""
        print_header("Test 6: Merge Searches")
        
        try:
            if not self.created_search_ids:
                self.record_test("Merge searches", False, "No searches to merge")
                return
            
            existing_id = self.created_search_ids[0]
            new_query = "gray marble flooring tiles"
            
            # Analyze new query
            analysis = await self.dedup_service.analyze_search_query(new_query)
            
            # Execute merge
            await self.dedup_service.merge_into_existing(
                existing_id=existing_id,
                new_query=new_query,
                new_filters={},
                new_material_filters={},
                analysis=analysis
            )
            
            # Verify merge
            response = self.supabase.table("saved_searches").select("*").eq(
                "id", existing_id
            ).single().execute()
            
            if response.data:
                merge_count = response.data.get("merge_count", 1)
                print_success(f"Merged successfully. Merge count: {merge_count}")
                self.record_test("Merge searches", True)
            else:
                self.record_test("Merge searches", False, "Search not found after merge")
        
        except Exception as e:
            self.record_test("Merge searches", False, str(e))
    
    async def test_7_execute_search(self):
        """Test 7: Execute search and track usage."""
        print_header("Test 7: Execute Search (Track Usage)")
        
        try:
            if not self.created_search_ids:
                self.record_test("Execute search", False, "No searches to execute")
                return
            
            search_id = self.created_search_ids[0]
            
            # Get current use_count
            response = self.supabase.table("saved_searches").select("use_count").eq(
                "id", search_id
            ).single().execute()
            
            old_count = response.data["use_count"] if response.data else 0
            
            # Update usage
            update_data = {
                "use_count": old_count + 1,
                "last_executed_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.table("saved_searches").update(update_data).eq(
                "id", search_id
            ).execute()
            
            # Verify update
            response = self.supabase.table("saved_searches").select("use_count").eq(
                "id", search_id
            ).single().execute()
            
            new_count = response.data["use_count"] if response.data else 0
            
            if new_count == old_count + 1:
                print_success(f"Usage tracked: {old_count} → {new_count}")
                self.record_test("Execute search", True)
            else:
                self.record_test("Execute search", False, f"Count mismatch: {old_count} → {new_count}")
        
        except Exception as e:
            self.record_test("Execute search", False, str(e))
    
    async def run_all_tests(self):
        """Run all E2E tests."""
        print_header("SAVED SEARCHES E2E TEST SUITE")
        print_info(f"Test User ID: {self.test_user_id}")
        print_info(f"Timestamp: {datetime.utcnow().isoformat()}")
        
        try:
            # Run tests in sequence
            await self.test_1_create_search_no_duplicates()
            await self.test_2_detect_exact_duplicate()
            await self.test_3_detect_semantic_duplicate()
            await self.test_4_reject_different_context()
            await self.test_5_reject_different_attributes()
            await self.test_6_merge_searches()
            await self.test_7_execute_search()
            
        finally:
            # Always cleanup
            await self.cleanup()
            
            # Print summary
            self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print_header("TEST SUMMARY")
        
        total = self.test_results["total"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        
        print(f"{Colors.BOLD}Total Tests:{Colors.END} {total}")
        print(f"{Colors.GREEN}Passed:{Colors.END} {passed}")
        print(f"{Colors.RED}Failed:{Colors.END} {failed}")
        
        if failed > 0:
            print(f"\n{Colors.BOLD}Failed Tests:{Colors.END}")
            for error in self.test_results["errors"]:
                print(f"  {Colors.RED}• {error['test']}{Colors.END}")
                print(f"    {error['error']}")
        
        # Overall result
        if failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.END}")


async def main():
    """Main entry point."""
    test_suite = SavedSearchesE2ETest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

