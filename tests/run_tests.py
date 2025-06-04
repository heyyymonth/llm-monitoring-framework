#!/usr/bin/env python3
"""
🧪 LLM Performance Monitoring Framework - Test Runner
========================================================
Comprehensive test suite runner with coverage reporting.
"""

import os
import sys
import subprocess
import glob
import argparse
from pathlib import Path

def print_header(title, char="="):
    """Print a formatted header"""
    print(f"\n{title}")
    print(char * len(title))

def print_success(message):
    """Print success message in green"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message in red"""
    print(f"❌ {message}")

def run_command(cmd, description=""):
    """Run a command and return success status"""
    if description:
        print(f"Running {description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False, e.stderr

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LLM monitoring framework tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--xml', action='store_true', help='Generate XML coverage report for CI')
    parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
    parser.add_argument('--fail-under', type=int, default=50, help='Fail if coverage is under this percentage')
    args = parser.parse_args()

    print_header("🧪 LLM Performance Monitoring Framework Tests")
    
    # Find test files
    test_files = glob.glob("tests/test_*.py")
    if not test_files:
        print_error("No test files found!")
        return False
    
    print(f"📁 Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"   • {test_file}")
    
    success = True
    
    # Run tests for each file
    for test_file in test_files:
        print_header(f"🧪 Running Tests: {test_file}")
        
        # Build pytest command with coverage if requested
        if args.coverage or args.xml or args.html:
            cmd = f"python -m pytest {test_file} -v"
            cmd += " --cov=monitoring --cov=api --cov=dashboard"
            cmd += f" --cov-fail-under={args.fail_under}"
            
            if args.xml:
                cmd += " --cov-report=xml"
            if args.html:
                cmd += " --cov-report=html"
            if not args.xml and not args.html:
                cmd += " --cov-report=term-missing"
        else:
            cmd = f"python -m pytest {test_file} -v"
        
        test_success, output = run_command(cmd, f"tests in {test_file}")
        print(output)
        
        if test_success:
            print_success(f"Tests passed in {test_file}")
        else:
            print_error(f"Tests failed in {test_file}")
            success = False
    
    # Run integration examples if available
    integration_examples = [
        "examples/integrations/test_ollama_llm.py",
        "examples/integrations/test_my_llm.py"
    ]
    
    available_examples = [ex for ex in integration_examples if os.path.exists(ex)]
    
    if available_examples:
        print_header("🧪 Integration Examples Available")
        print("🔧 Run integration examples separately:")
        for example in available_examples:
            print(f"   python {example}")
    
    # Run quick test
    quick_test_path = "examples/integrations/quick_test.py"
    if os.path.exists(quick_test_path):
        print_header("🧪 Quick Functionality Test")
        print(f"Running quick test: python {quick_test_path}")
        print("-" * 60)
        
        test_success, output = run_command(f"python {quick_test_path}", "quick functionality test")
        print(output)
        
        if test_success:
            print_success("Quick test passed")
        else:
            print_error("Quick test failed")
            success = False
    
    # Print summary
    print_header("🧪 Test Summary")
    if success:
        print_success("All tests passed!")
        if args.coverage or args.html:
            print("📊 Coverage report generated in htmlcov/index.html")
        if args.xml:
            print("📊 XML coverage report generated as coverage.xml")
    else:
        print_error("Some tests failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 