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
        print(f"Running {description}: {cmd}")
        print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Failed to run command: {e}")
        return False

def find_test_files():
    """Find all test files"""
    test_files = []
    
    # Main test file
    main_test = "tests/test_monitoring.py"
    if os.path.exists(main_test):
        test_files.append(main_test)
    
    return test_files

def run_tests():
    """Run the test suite"""
    print_header("🧪 LLM Performance Monitoring Framework Tests", "=")
    
    # Find test files
    test_files = find_test_files()
    
    if not test_files:
        print_error("No test files found!")
        return False
    
    print(f"📁 Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"   • {test_file}")
    
    all_passed = True
    
    # Run main test suite
    for test_file in test_files:
        print_header(f"🧪 Running Tests: {test_file}", "=")
        
        cmd = (f"python -m pytest {test_file} -v --tb=short --durations=10 "
               f"--cov=monitoring --cov=api --cov=dashboard "
               f"--cov-report=term-missing --cov-report=html:htmlcov")
        
        if run_command(cmd):
            print_success(f"Tests passed in {test_file}")
        else:
            print_error(f"Tests failed in {test_file}")
            all_passed = False
    
    # Run integration examples
    print_header("🧪 Integration Examples Available", "=")
    
    integration_examples = [
        "examples/integrations/test_ollama_llm.py",
        "examples/integrations/test_my_llm.py"
    ]
    
    available_examples = [ex for ex in integration_examples if os.path.exists(ex)]
    
    if available_examples:
        print("🔧 Run integration examples separately:")
        for example in available_examples:
            print(f"   python {example}")
    
    # Run quick test
    print_header("🧪 Quick Functionality Test", "=")
    quick_test_path = "examples/integrations/quick_test.py"
    
    if os.path.exists(quick_test_path):
        print(f"Running quick test: python {quick_test_path}")
        print("-" * 60)
        if run_command(f"python {quick_test_path}"):
            print_success("Quick test passed")
        else:
            print_error("Quick test failed")
            all_passed = False
    
    # Summary
    print_header("🧪 Test Summary", "=")
    
    if all_passed:
        print_success("All tests passed!")
        if os.path.exists("htmlcov/index.html"):
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print_error("Some tests failed!")
        return False
    
    return True

def main():
    """Main function"""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 