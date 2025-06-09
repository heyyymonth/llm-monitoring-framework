#!/usr/bin/env python3
"""
Quality monitoring component validation script for CI testing.
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

def test_quality_components():
    """Test quality monitoring components with error handling."""
    try:
        print("🎯 Testing quality monitoring imports...")
        from monitoring.quality import QualityAssessor, SafetyEvaluator, HallucinationDetector
        from monitoring.models import QualityMetrics, SafetyAssessment
        print("  ✅ Quality monitoring imports successful")
        
        # Test quality assessor
        print("📊 Testing QualityAssessor...")
        assessor = QualityAssessor()
        print('  ✅ QualityAssessor initialized')
        
        # Test safety evaluator  
        print("🛡️  Testing SafetyEvaluator...")
        evaluator = SafetyEvaluator()
        print('  ✅ SafetyEvaluator initialized')
        
        # Test hallucination detector
        print("🔍 Testing HallucinationDetector...")
        detector = HallucinationDetector()
        print('  ✅ HallucinationDetector initialized')
        
        print('\n🎉 Quality monitoring components working correctly!')
        return True
        
    except Exception as e:
        print(f"❌ Quality components test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🔍 Starting quality components test...\n")
    
    success = test_quality_components()
    
    if success:
        print("\n✅ Quality components test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Quality components test failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 