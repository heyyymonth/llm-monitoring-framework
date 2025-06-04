#!/usr/bin/env python3
"""
Test runner for the LLM Performance Monitoring Framework.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print(f"\n🧪 {title}")
    print("=" * 60)

def print_separator():
    """Print a separator line."""
    print("-" * 60)

def run_command(cmd, description="Running command"):
    """Run a command and capture its output."""
    print(f"{description}: {' '.join(cmd)}")
    print_separator()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=False, 
            text=True, 
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def find_test_files():
    """Find all test files in the project."""
    project_root = Path(__file__).parent.parent
    test_files = []
    
    # Look for test files in tests directory
    tests_dir = project_root / "tests"
    if tests_dir.exists():
        test_files.extend(list(tests_dir.glob("test_*.py")))
        test_files.extend(list(tests_dir.glob("**/test_*.py")))
    
    return [str(f.relative_to(project_root)) for f in test_files if f.exists()]

def main():
    """Run the complete test suite."""
    print_header("LLM Performance Monitoring Framework Tests")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Find test files
    test_files = find_test_files()
    
    if not test_files:
        print("❌ No test files found!")
        print("Looking for test files in:")
        print("  - tests/test_*.py")
        print("  - tests/**/test_*.py")
        return False
    
    print(f"📁 Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"   • {test_file}")
    
    # Run tests with pytest
    success = True
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  Skipping missing file: {test_file}")
            continue
            
        print_header(f"Running Tests: {test_file}")
        
        cmd = [
            "python", "-m", "pytest", 
            test_file,
            "-v",
            "--tb=short",
            "--durations=10",
            "--cov=monitoring",
            "--cov=api", 
            "--cov=dashboard",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ]
        
        if not run_command(cmd, "Running pytest"):
            success = False
            print(f"❌ Tests failed in {test_file}")
        else:
            print(f"✅ Tests passed in {test_file}")
    
    # Run integration tests if they exist
    integration_tests = list(Path("examples/integrations").glob("test_*.py"))
    if integration_tests:
        print_header("Integration Tests Available")
        print("🔧 Run integration tests separately:")
        for test in integration_tests:
            print(f"   python {test}")
    
    # Run quick test if available
    quick_test = Path("examples/integrations/quick_test.py")
    if quick_test.exists():
        print_header("Quick Functionality Test")
        if run_command(["python", str(quick_test)], "Running quick test"):
            print("✅ Quick test passed")
        else:
            print("❌ Quick test failed")
            success = False
    
    print_header("Test Summary")
    
    if success:
        print("🎉 All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        
        # Show coverage summary if available
        try:
            result = subprocess.run(
                ["coverage", "report", "--show-missing"], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print("\n📈 Coverage Summary:")
                print(result.stdout)
        except:
            pass
            
        return True
    else:
        print("❌ Some tests failed!")
        print("💡 Check the output above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 