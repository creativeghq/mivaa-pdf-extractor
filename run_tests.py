#!/usr/bin/env python3
"""
MIVAA PDF Extractor Test Runner

This script provides a comprehensive test runner for all test suites:
- Unit tests
- Integration tests  
- End-to-end tests
- Performance tests

Usage:
    python run_tests.py [options]
    
Options:
    --unit          Run only unit tests
    --integration   Run only integration tests
    --e2e           Run only end-to-end tests
    --performance   Run only performance tests
    --coverage      Generate coverage report
    --verbose       Verbose output
    --fast          Skip slow tests
    --help          Show this help message
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any


class TestRunner:
    """Comprehensive test runner for MIVAA PDF Extractor."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results: Dict[str, Any] = {}
        
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            print(f"Exit code: {result.returncode}")
            print(f"Execution time: {execution_time:.2f}s")
            
            if result.stdout:
                print(f"\nSTDOUT:\n{result.stdout}")
            
            if result.stderr:
                print(f"\nSTDERR:\n{result.stderr}")
            
            success = result.returncode == 0
            
            self.test_results[description] = {
                'success': success,
                'exit_code': result.returncode,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            return success
            
        except subprocess.TimeoutExpired:
            print(f"❌ Command timed out after 5 minutes")
            self.test_results[description] = {
                'success': False,
                'exit_code': -1,
                'execution_time': 300,
                'stdout': '',
                'stderr': 'Command timed out'
            }
            return False
            
        except Exception as e:
            print(f"❌ Error running command: {e}")
            self.test_results[description] = {
                'success': False,
                'exit_code': -1,
                'execution_time': 0,
                'stdout': '',
                'stderr': str(e)
            }
            return False
    
    def run_unit_tests(self, verbose: bool = False, fast: bool = False) -> bool:
        """Run unit tests."""
        command = ["python", "-m", "pytest", "tests/unit/"]
        
        if verbose:
            command.append("-v")
        
        if fast:
            command.extend(["-m", "not slow"])
        
        command.extend([
            "--tb=short",
            "--durations=10"
        ])
        
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self, verbose: bool = False, fast: bool = False) -> bool:
        """Run integration tests."""
        command = ["python", "-m", "pytest", "tests/integration/"]
        
        if verbose:
            command.append("-v")
        
        if fast:
            command.extend(["-m", "not slow"])
        
        command.extend([
            "--tb=short",
            "--durations=10"
        ])
        
        return self.run_command(command, "Integration Tests")
    
    def run_e2e_tests(self, verbose: bool = False, fast: bool = False) -> bool:
        """Run end-to-end tests."""
        command = ["python", "-m", "pytest", "tests/e2e/"]
        
        if verbose:
            command.append("-v")
        
        if fast:
            command.extend(["-m", "not slow"])
        
        command.extend([
            "--tb=short",
            "--durations=10"
        ])
        
        return self.run_command(command, "End-to-End Tests")
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests."""
        command = ["python", "-m", "pytest", "tests/performance/"]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=10",
            "-m", "performance"
        ])
        
        return self.run_command(command, "Performance Tests")
    
    def run_coverage_tests(self, verbose: bool = False) -> bool:
        """Run tests with coverage reporting."""
        command = [
            "python", "-m", "pytest",
            "--cov=app",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
            "tests/"
        ]
        
        if verbose:
            command.append("-v")
        
        command.extend([
            "--tb=short",
            "--durations=10"
        ])
        
        return self.run_command(command, "Coverage Tests")
    
    def run_linting(self) -> bool:
        """Run code linting checks."""
        commands = [
            (["python", "-m", "flake8", "app/", "tests/"], "Flake8 Linting"),
            (["python", "-m", "black", "--check", "app/", "tests/"], "Black Code Formatting Check"),
            (["python", "-m", "isort", "--check-only", "app/", "tests/"], "Import Sorting Check"),
            (["python", "-m", "mypy", "app/"], "Type Checking")
        ]
        
        all_passed = True
        for command, description in commands:
            try:
                success = self.run_command(command, description)
                if not success:
                    all_passed = False
            except Exception:
                # Some linting tools might not be installed
                print(f"⚠️  Skipping {description} (tool not available)")
                continue
        
        return all_passed
    
    def run_security_checks(self) -> bool:
        """Run security checks."""
        commands = [
            (["python", "-m", "bandit", "-r", "app/"], "Security Check (Bandit)"),
            (["python", "-m", "safety", "check"], "Dependency Security Check")
        ]
        
        all_passed = True
        for command, description in commands:
            try:
                success = self.run_command(command, description)
                if not success:
                    all_passed = False
            except Exception:
                print(f"⚠️  Skipping {description} (tool not available)")
                continue
        
        return all_passed
    
    def print_summary(self):
        """Print test execution summary."""
        print(f"\n{'='*80}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total test suites: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        
        total_time = sum(result['execution_time'] for result in self.test_results.values())
        print(f"Total execution time: {total_time:.2f}s")
        
        print(f"\nDetailed Results:")
        print(f"{'-'*80}")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_str = f"{result['execution_time']:.2f}s"
            print(f"{status:<8} {test_name:<40} {time_str:>8}")
        
        if failed_tests > 0:
            print(f"\n❌ {failed_tests} test suite(s) failed!")
            print("Check the detailed output above for error information.")
            return False
        else:
            print(f"\n✅ All {passed_tests} test suite(s) passed!")
            return True
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        print("Checking test dependencies...")
        
        required_packages = [
            "pytest",
            "pytest-asyncio", 
            "pytest-cov",
            "pytest-mock",
            "factory-boy",
            "faker",
            "httpx",
            "psutil"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install them with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All required dependencies are installed")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MIVAA PDF Extractor Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Check dependencies first
    if not runner.check_dependencies():
        if args.check_deps:
            return 1
        print("\n⚠️  Some dependencies are missing, but continuing with tests...")
    
    if args.check_deps:
        return 0
    
    print(f"\n🚀 Starting MIVAA PDF Extractor Test Suite")
    print(f"Project root: {runner.project_root}")
    
    all_passed = True
    
    # Determine which tests to run
    run_specific = any([args.unit, args.integration, args.e2e, args.performance, args.coverage, args.lint, args.security])
    
    if args.unit or not run_specific:
        if not runner.run_unit_tests(args.verbose, args.fast):
            all_passed = False
    
    if args.integration or not run_specific:
        if not runner.run_integration_tests(args.verbose, args.fast):
            all_passed = False
    
    if args.e2e or not run_specific:
        if not runner.run_e2e_tests(args.verbose, args.fast):
            all_passed = False
    
    if args.performance:
        if not runner.run_performance_tests(args.verbose):
            all_passed = False
    
    if args.coverage:
        if not runner.run_coverage_tests(args.verbose):
            all_passed = False
    
    if args.lint:
        if not runner.run_linting():
            all_passed = False
    
    if args.security:
        if not runner.run_security_checks():
            all_passed = False
    
    # Print summary
    summary_passed = runner.print_summary()
    
    if all_passed and summary_passed:
        print(f"\n🎉 All tests completed successfully!")
        return 0
    else:
        print(f"\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())