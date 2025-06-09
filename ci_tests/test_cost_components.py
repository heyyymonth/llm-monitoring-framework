#!/usr/bin/env python3
"""
Cost tracking component validation script for CI testing.
Enhanced with error handling and diagnostics.
"""

import sys
import os
import traceback

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"🐍 Python: {sys.version}")
print(f"🖥️  Platform: {sys.platform}")
print(f"📁 Working dir: {os.getcwd()}")

def test_cost_components():
    """Test cost tracking components with error handling."""
    try:
        print("💰 Testing cost tracking imports...")
        from monitoring.cost import CostTracker, BudgetManager
        from monitoring.models import CostMetrics
        print("  ✅ Cost tracking imports successful")
        
        # Test cost tracker
        print("📊 Testing CostTracker...")
        tracker = CostTracker()
        print('  ✅ CostTracker initialized')
        
        # Test budget manager
        print("🎯 Testing BudgetManager...")
        manager = BudgetManager()
        print('  ✅ BudgetManager initialized')
        
        print('\n🎉 Cost tracking components working correctly!')
        return True
        
    except Exception as e:
        print(f"❌ Cost components test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🔍 Starting cost components test...\n")
    
    success = test_cost_components()
    
    if success:
        print("\n✅ Cost components test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Cost components test failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 