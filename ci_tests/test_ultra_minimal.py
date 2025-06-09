#!/usr/bin/env python3
"""
Ultra-minimal test to isolate CI import issues.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root in sys.path}")

# Test 1: Basic module discovery
try:
    import monitoring
    print("✅ monitoring package found")
except ImportError as e:
    print(f"❌ monitoring package not found: {e}")
    sys.exit(1)

try:
    import api
    print("✅ api package found")
except ImportError as e:
    print(f"❌ api package not found: {e}")
    sys.exit(1)

# Test 2: Specific module imports (without instantiation)
try:
    from monitoring import models
    print("✅ monitoring.models imported")
except ImportError as e:
    print(f"❌ monitoring.models failed: {e}")
    sys.exit(1)

# Test 3: Check if we can import one simple class
try:
    from monitoring.models import LLMTrace
    print("✅ LLMTrace class imported")
except ImportError as e:
    print(f"❌ LLMTrace import failed: {e}")
    sys.exit(1)

print("🎉 All ultra-minimal tests passed!") 