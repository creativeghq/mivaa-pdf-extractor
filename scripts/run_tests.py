#!/usr/bin/env python3
"""
Comprehensive test execution script for MIVAA PDF Extractor.

This script provides a unified interface for running different types of tests
with coverage reporting, parallel execution, and detailed reporting.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional
import json
import time
from datetime import datetime


class TestRunner:
    """Comprehensive test runner with coverage and reporting capabilities."""
    
    def __init__(self, project_root: str = None):
        """Initialize the test runner."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "htmlcov"
        self.reports_dir = self.project_root / "test_reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
        
    def run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"Error running command: {e}")
            return subprocess.CompletedProcess(command, 1, "", str(e))
    
    def run_unit_tests(self, verbose: bool = False, parallel: bool = False) -> bool:
        """Run unit tests with coverage."""
        print("\n🧪 Running Unit Tests...")
        
        command = [
            "python", "-m", "pytest",
            "tests/unit/",
            "--cov=app",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-config=.coveragerc",
            "--junit-xml=test_reports/unit_tests.xml",
            "-v" if verbose else "-q"
        ]
        
        if parallel:
            command.extend(["-n", "auto"])
        
        result = self.run_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Unit tests passed!")
            return True
        else:
            print("❌ Unit tests failed!")
            return False
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests...")
        
        command = [
            "python", "-m", "pytest",
            "tests/integration/",
            "--junit-xml=test_reports/integration_tests.xml",
            "-v" if verbose else "-q",
            "-m", "integration"
        ]
        
        result = self.run_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Integration tests passed!")
            return True
        else:
            print("❌ Integration tests failed!")
            return False
    
    def run_e2e_tests(self, verbose: bool = False) -> bool:
        """Run end-to-end tests."""
        print("\n🌐 Running End-to-End Tests...")
        
        command = [
            "python", "-m", "pytest",
            "tests/e2e/",
            "--junit-xml=test_reports/e2e_tests.xml",
            "-v" if verbose else "-q",
            "-m", "e2e"
        ]
        
        result = self.run_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ E2E tests passed!")
            return True
        else:
            print("❌ E2E tests failed!")
            return False
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests."""
        print("\n⚡ Running Performance Tests...")
        
        command = [
            "python", "-m", "pytest",
            "tests/performance/",
            "--junit-xml=test_reports/performance_tests.xml",
            "-v" if verbose else "-q",
            "-m", "performance"
        ]
        
        result = self.run_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Performance tests passed!")
            return True
        else:
            print("❌ Performance tests failed!")
            return False
    
    def run_security_tests(self, verbose: bool = False) -> bool:
        """Run security tests."""
        print("\n🔒 Running Security Tests...")
        
        command = [
            "python", "-m", "pytest",
            "tests/security/",
            "--junit-xml=test_reports/security_tests.xml",
            "-v" if verbose else "-q",
            "-m", "security"
        ]
        
        result = self.run_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Security tests passed!")
            return True
        else:
            print("❌ Security tests failed!")
            return False
    
    def run_all_tests(self, verbose: bool = False, parallel: bool = False) -> bool:
        """Run all test suites."""
        print("\n🚀 Running All Tests...")
        
        results = []
        
        # Run unit tests with coverage
        results.append(self.run_unit_tests(verbose, parallel))
        
        # Run integration tests
        if self.test_dir.joinpath("integration").exists():
            results.append(self.run_integration_tests(verbose))
        
        # Run E2E tests
        if self.test_dir.joinpath("e2e").exists():
            results.append(self.run_e2e_tests(verbose))
        
        # Run performance tests
        if self.test_dir.joinpath("performance").exists():
            results.append(self.run_performance_tests(verbose))
        
        # Run security tests
        if self.test_dir.joinpath("security").exists():
            results.append(self.run_security_tests(verbose))
        
        all_passed = all(results)
        
        if all_passed:
            print("\n🎉 All tests passed!")
        else:
            print("\n💥 Some tests failed!")
        
        return all_passed
    
    def check_coverage_threshold(self, threshold: float = 90.0) -> bool:
        """Check if coverage meets the minimum threshold."""
        coverage_file = self.project_root / "coverage.xml"
        
        if not coverage_file.exists():
            print(f"❌ Coverage file not found: {coverage_file}")
            return False
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Find coverage percentage
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get("line-rate", 0)) * 100
                
                print(f"\n📊 Code Coverage: {line_rate:.2f}%")
                
                if line_rate >= threshold:
                    print(f"✅ Coverage meets threshold ({threshold}%)")
                    return True
                else:
                    print(f"❌ Coverage below threshold ({threshold}%)")
                    return False
            else:
                print("❌ Could not parse coverage data")
                return False
                
        except Exception as e:
            print(f"❌ Error checking coverage: {e}")
            return False
    
    def generate_test_report(self) -> None:
        """Generate a comprehensive test report."""
        print("\n📋 Generating Test Report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": "MIVAA PDF Extractor",
            "test_results": {},
            "coverage": {},
            "summary": {}
        }
        
        # Check for test result files
        test_files = {
            "unit": "unit_tests.xml",
            "integration": "integration_tests.xml",
            "e2e": "e2e_tests.xml",
            "performance": "performance_tests.xml",
            "security": "security_tests.xml"
        }
        
        total_tests = 0
        total_failures = 0
        
        for test_type, filename in test_files.items():
            filepath = self.reports_dir / filename
            if filepath.exists():
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    
                    tests = int(root.get("tests", 0))
                    failures = int(root.get("failures", 0))
                    errors = int(root.get("errors", 0))
                    time_taken = float(root.get("time", 0))
                    
                    report["test_results"][test_type] = {
                        "tests": tests,
                        "failures": failures,
                        "errors": errors,
                        "time": time_taken,
                        "success_rate": ((tests - failures - errors) / tests * 100) if tests > 0 else 0
                    }
                    
                    total_tests += tests
                    total_failures += failures + errors
                    
                except Exception as e:
                    print(f"Warning: Could not parse {filename}: {e}")
        
        # Add coverage information
        coverage_file = self.project_root / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    report["coverage"] = {
                        "line_rate": float(coverage_elem.get("line-rate", 0)) * 100,
                        "branch_rate": float(coverage_elem.get("branch-rate", 0)) * 100,
                        "lines_covered": int(coverage_elem.get("lines-covered", 0)),
                        "lines_valid": int(coverage_elem.get("lines-valid", 0))
                    }
            except Exception as e:
                print(f"Warning: Could not parse coverage data: {e}")
        
        # Add summary
        report["summary"] = {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "overall_success_rate": ((total_tests - total_failures) / total_tests * 100) if total_tests > 0 else 0,
            "status": "PASSED" if total_failures == 0 else "FAILED"
        }
        
        # Save report
        report_file = self.reports_dir / "test_summary.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {report_file}")
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Failures: {total_failures}")
        print(f"   Success Rate: {report['summary']['overall_success_rate']:.2f}%")
        if "coverage" in report:
            print(f"   Code Coverage: {report['coverage']['line_rate']:.2f}%")
    
    def clean_reports(self) -> None:
        """Clean up old test reports and coverage data."""
        print("\n🧹 Cleaning up old reports...")
        
        # Remove coverage files
        for pattern in ["htmlcov", "coverage.xml", ".coverage"]:
            path = self.project_root / pattern
            if path.exists():
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:
                    path.unlink()
        
        # Remove test reports
        if self.reports_dir.exists():
            for file in self.reports_dir.glob("*.xml"):
                file.unlink()
            for file in self.reports_dir.glob("*.json"):
                file.unlink()
        
        print("✅ Cleanup completed!")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="MIVAA PDF Extractor Test Runner")
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "e2e", "performance", "security", "all"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel (unit tests only)"
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=90.0,
        help="Minimum coverage threshold (default: 90%%)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up old reports before running tests"
    )
    
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate test report from existing results"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Clean up if requested
    if args.clean:
        runner.clean_reports()
    
    # Generate report only if requested
    if args.report_only:
        runner.generate_test_report()
        return
    
    # Run tests
    success = False
    
    if args.test_type == "unit":
        success = runner.run_unit_tests(args.verbose, args.parallel)
    elif args.test_type == "integration":
        success = runner.run_integration_tests(args.verbose)
    elif args.test_type == "e2e":
        success = runner.run_e2e_tests(args.verbose)
    elif args.test_type == "performance":
        success = runner.run_performance_tests(args.verbose)
    elif args.test_type == "security":
        success = runner.run_security_tests(args.verbose)
    elif args.test_type == "all":
        success = runner.run_all_tests(args.verbose, args.parallel)
    
    # Check coverage threshold for unit tests
    coverage_ok = True
    if args.test_type in ["unit", "all"]:
        coverage_ok = runner.check_coverage_threshold(args.coverage_threshold)
    
    # Generate comprehensive report
    runner.generate_test_report()
    
    # Exit with appropriate code
    if success and coverage_ok:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed or coverage below threshold!")
        sys.exit(1)


if __name__ == "__main__":
    main()