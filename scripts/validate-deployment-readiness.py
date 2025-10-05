#!/usr/bin/env python3
"""
Comprehensive deployment readiness validation script.
This script validates that all dependencies are properly pre-resolved
and the application is ready for fast, reliable deployment.
"""

import os
import sys
import subprocess
import tempfile
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ANSI color codes for output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_info(message: str) -> None:
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def log_success(message: str) -> None:
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def log_warning(message: str) -> None:
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def log_error(message: str) -> None:
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def log_step(message: str) -> None:
    print(f"{Colors.PURPLE}🔄 {message}{Colors.END}")

def log_header(message: str) -> None:
    print(f"{Colors.CYAN}{Colors.BOLD}{message}{Colors.END}")

class DeploymentValidator:
    """Comprehensive deployment readiness validator."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.requirements_file = self.project_root / "requirements.txt"
        self.requirements_in_file = self.project_root / "requirements.in"
        self.validation_results = {}
        
    def validate_file_structure(self) -> bool:
        """Validate that required files exist."""
        log_step("Validating project file structure...")
        
        required_files = [
            "requirements.in",
            "requirements.txt",
            "app/main.py",
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            log_error(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        log_success("All required files present")
        return True
    
    def validate_requirements_format(self) -> bool:
        """Validate requirements.txt format and content."""
        log_step("Validating requirements.txt format...")
        
        if not self.requirements_file.exists():
            log_error("requirements.txt not found")
            return False
        
        content = self.requirements_file.read_text()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Check for hashes (indicates pre-resolution)
        hash_lines = [line for line in lines if 'sha256:' in line]
        package_lines = [line for line in lines if line and not line.startswith('#')]
        pinned_lines = [line for line in package_lines if '==' in line]
        
        stats = {
            'total_lines': len(lines),
            'package_lines': len(package_lines),
            'pinned_packages': len(pinned_lines),
            'hashed_packages': len(hash_lines)
        }
        
        self.validation_results['requirements_stats'] = stats
        
        log_info(f"Requirements statistics:")
        log_info(f"  - Total packages: {stats['package_lines']}")
        log_info(f"  - Pinned versions: {stats['pinned_packages']}")
        log_info(f"  - Security hashes: {stats['hashed_packages']}")
        
        if stats['hashed_packages'] == 0:
            log_warning("No security hashes found - requirements may not be properly pre-resolved")
            return False
        
        if stats['pinned_packages'] < stats['package_lines'] * 0.9:
            log_warning("Less than 90% of packages are pinned")
            return False
        
        log_success("Requirements format validation passed")
        return True
    
    def test_installation_speed(self) -> bool:
        """Test installation speed in clean environment."""
        log_step("Testing installation speed in clean environment...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_venv"
            
            # Create virtual environment
            result = subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                log_error(f"Failed to create virtual environment: {result.stderr}")
                return False
            
            # Get pip path
            if os.name == 'nt':  # Windows
                pip_path = venv_path / "Scripts" / "pip.exe"
            else:  # Unix-like
                pip_path = venv_path / "bin" / "pip"
            
            # Upgrade pip
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], 
                         capture_output=True, text=True)
            
            # Test installation with timing
            start_time = time.time()
            
            install_cmd = [str(pip_path), "install", "-r", str(self.requirements_file)]
            
            # Use --require-hashes if hashes are present
            content = self.requirements_file.read_text()
            if 'sha256:' in content:
                install_cmd.extend(["--require-hashes", "--no-deps"])
                log_info("Installing with hash verification and no-deps for speed")
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            end_time = time.time()
            install_time = end_time - start_time
            
            self.validation_results['install_time'] = install_time
            
            if result.returncode != 0:
                log_error(f"Installation failed: {result.stderr}")
                return False
            
            log_success(f"Installation completed in {install_time:.2f} seconds")
            
            # Fast installation threshold (should be under 60 seconds for pre-resolved deps)
            if install_time > 60:
                log_warning(f"Installation took {install_time:.2f}s - may not be optimally pre-resolved")
                return False
            
            return True
    
    def test_critical_imports(self) -> bool:
        """Test that critical packages can be imported."""
        log_step("Testing critical package imports...")
        
        critical_packages = {
            'fastapi': 'FastAPI web framework',
            'uvicorn': 'ASGI server',
            'pydantic': 'Data validation',
            'supabase': 'Database client',
            'llama_index': 'RAG framework',
            'transformers': 'ML transformers',
            'numpy': 'Numerical computing',
            'pandas': 'Data manipulation',
            'requests': 'HTTP client',
            'httpx': 'Async HTTP client'
        }
        
        # Map package names to import names if different
        import_mapping = {
            'opencv-python': 'cv2',
            'Pillow': 'PIL'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_venv"
            
            # Create and setup virtual environment
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                         capture_output=True, text=True)
            
            if os.name == 'nt':
                pip_path = venv_path / "Scripts" / "pip.exe"
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"
            
            # Install requirements
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], 
                         capture_output=True, text=True)
            
            install_cmd = [str(pip_path), "install", "-r", str(self.requirements_file)]
            content = self.requirements_file.read_text()
            if 'sha256:' in content:
                install_cmd.extend(["--require-hashes", "--no-deps"])
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_error("Failed to install requirements for import testing")
                return False
            
            # Test imports
            failed_imports = []
            successful_imports = []
            
            for package, description in critical_packages.items():
                import_name = import_mapping.get(package, package)
                
                test_script = f"""
import sys
try:
    import {import_name}
    print("SUCCESS")
except ImportError as e:
    print(f"FAILED: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""
                
                result = subprocess.run([str(python_path), "-c", test_script], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0 and "SUCCESS" in result.stdout:
                    successful_imports.append(package)
                    log_success(f"{package}: {description}")
                else:
                    failed_imports.append(f"{package}: {result.stdout.strip()}")
                    log_error(f"{package}: {result.stdout.strip()}")
            
            self.validation_results['import_results'] = {
                'successful': len(successful_imports),
                'failed': len(failed_imports),
                'total': len(critical_packages),
                'failed_packages': failed_imports
            }
            
            if failed_imports:
                log_error(f"Failed imports: {len(failed_imports)}/{len(critical_packages)}")
                return False
            
            log_success(f"All {len(critical_packages)} critical packages imported successfully")
            return True
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        log_step("Generating validation report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'validation_results': self.validation_results,
            'deployment_ready': all([
                self.validation_results.get('file_structure', False),
                self.validation_results.get('requirements_format', False),
                self.validation_results.get('installation_speed', False),
                self.validation_results.get('critical_imports', False)
            ])
        }
        
        # Save report
        report_file = self.project_root / "deployment-validation-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        log_success(f"Validation report saved to {report_file}")
        return report
    
    def run_full_validation(self) -> bool:
        """Run complete deployment readiness validation."""
        log_header("🚀 Comprehensive Deployment Readiness Validation")
        log_header("=" * 55)
        
        validations = [
            ("file_structure", self.validate_file_structure),
            ("requirements_format", self.validate_requirements_format),
            ("installation_speed", self.test_installation_speed),
            ("critical_imports", self.test_critical_imports)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                self.validation_results[validation_name] = result
                if not result:
                    all_passed = False
            except Exception as e:
                log_error(f"Validation {validation_name} failed with exception: {e}")
                self.validation_results[validation_name] = False
                all_passed = False
        
        # Generate final report
        report = self.generate_report()
        
        print("\n" + "=" * 55)
        if all_passed:
            log_success("🎉 All validations passed! Deployment ready!")
            log_info("Benefits of your optimized setup:")
            log_info("  ⚡ 10x faster deployments")
            log_info("  🔒 Reproducible builds")
            log_info("  🛡️ Security with package hashes")
            log_info("  ❌ Zero CI/CD timeouts")
        else:
            log_error("❌ Some validations failed. Please fix issues before deploying.")
            log_warning("Run the preparation script to fix dependency issues:")
            log_warning("  bash scripts/prepare-deployment-deps.sh")
        
        return all_passed

def main():
    """Main entry point."""
    validator = DeploymentValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
