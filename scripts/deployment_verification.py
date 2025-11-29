#!/usr/bin/env python3
"""
DEPLOYMENT VERIFICATION SCRIPT
Comprehensive package installation and verification with visual progress screens
"""

import subprocess
import sys
import importlib
from typing import Dict, List, Tuple
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Critical packages that must be installed
CRITICAL_PACKAGES = [
    ('fastapi', 'Web Framework', 'FastAPI web framework'),
    ('uvicorn', 'Server', 'ASGI server'),
    ('cv2', 'Computer Vision', 'OpenCV image processing'),
    ('fitz', 'PDF Processing', 'PyMuPDF PDF library'),
    ('pymupdf4llm', 'PDF Processing', 'PDF processing for LLM'),
    ('supabase', 'Database', 'Supabase client'),
    ('httpx', 'HTTP Client', 'Async HTTP client'),
    ('anthropic', 'AI', 'Anthropic Claude API'),
    ('openai', 'AI', 'OpenAI GPT API'),
    ('together', 'AI', 'Together AI API'),
    ('pandas', 'Data Processing', 'Data analysis library'),
    ('numpy', 'Numerical', 'Numerical computing'),
    ('PIL', 'Image Processing', 'Pillow image library'),
    ('pydantic', 'Validation', 'Data validation'),
    ('requests', 'HTTP Client', 'HTTP library'),
    ('sentry_sdk', 'Monitoring', 'Sentry error tracking'),
]


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{title.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{BOLD}{YELLOW}{'─' * 80}{RESET}")
    print(f"{BOLD}{YELLOW}{title}{RESET}")
    print(f"{BOLD}{YELLOW}{'─' * 80}{RESET}\n")


def check_package(package_name: str, category: str, description: str) -> Tuple[bool, str]:
    """Check if a package is installed and get its version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def verify_all_packages() -> Dict[str, Dict]:
    """Verify all critical packages and return status."""
    print_section("📦 VERIFYING CRITICAL PACKAGES")
    
    results = {}
    total = len(CRITICAL_PACKAGES)
    installed = 0
    
    for i, (package, category, description) in enumerate(CRITICAL_PACKAGES, 1):
        # Progress indicator
        progress = f"[{i}/{total}]"
        print(f"{progress} Checking {package}...", end=' ')
        
        is_installed, version = check_package(package, category, description)
        
        if is_installed:
            print(f"{GREEN}✓ OK{RESET} (v{version})")
            installed += 1
            status = 'installed'
        else:
            print(f"{RED}✗ MISSING{RESET}")
            status = 'missing'
        
        results[package] = {
            'category': category,
            'description': description,
            'status': status,
            'version': version if is_installed else 'N/A'
        }
    
    # Summary
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  • Total packages: {total}")
    print(f"  • Installed: {GREEN}{installed}{RESET}")
    print(f"  • Missing: {RED}{total - installed}{RESET}")
    print(f"  • Success rate: {GREEN}{(installed/total)*100:.1f}%{RESET}")
    
    return results


def generate_markdown_report(results: Dict[str, Dict]) -> str:
    """Generate a markdown report for GitHub Actions summary."""
    lines = []
    lines.append("## 📦 Package Installation Status")
    lines.append("")
    lines.append("| Package | Category | Description | Status | Version |")
    lines.append("|---------|----------|-------------|--------|---------|")
    
    for package, info in results.items():
        status_icon = "🟢" if info['status'] == 'installed' else "🔴"
        status_text = "Installed" if info['status'] == 'installed' else "Missing"
        lines.append(
            f"| **{package}** | {info['category']} | {info['description']} | "
            f"{status_icon} {status_text} | {info['version']} |"
        )
    
    lines.append("")
    
    # Add summary statistics
    total = len(results)
    installed = sum(1 for r in results.values() if r['status'] == 'installed')
    success_rate = (installed / total) * 100
    
    lines.append("### 📊 Installation Summary")
    lines.append("")
    lines.append(f"- **Total Packages**: {total}")
    lines.append(f"- **Successfully Installed**: {installed}")
    lines.append(f"- **Missing**: {total - installed}")
    lines.append(f"- **Success Rate**: {success_rate:.1f}%")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main verification function."""
    print_header("🚀 MIVAA DEPLOYMENT VERIFICATION")
    
    print(f"{BOLD}Deployment Information:{RESET}")
    print(f"  • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  • Python Version: {sys.version.split()[0]}")
    print(f"  • Platform: {sys.platform}")
    
    # Verify packages
    results = verify_all_packages()
    
    # Generate markdown report
    markdown = generate_markdown_report(results)
    
    # Save to file for GitHub Actions
    with open('/tmp/package_status_report.md', 'w') as f:
        f.write(markdown)
    
    print_section("✅ VERIFICATION COMPLETE")
    print(f"Report saved to: /tmp/package_status_report.md")
    
    # Exit with error if any packages are missing
    missing = sum(1 for r in results.values() if r['status'] == 'missing')
    if missing > 0:
        print(f"\n{RED}{BOLD}⚠️  WARNING: {missing} package(s) missing!{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}✅ All critical packages installed successfully!{RESET}")
        sys.exit(0)


if __name__ == '__main__':
    main()

