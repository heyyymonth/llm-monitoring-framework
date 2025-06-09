#!/usr/bin/env python3
"""
API server validation script for CI testing.
"""

import sys
import traceback

def main():
    try:
        print(f"🐍 Python version: {sys.version}")
        
        # Test individual monitoring imports first
        print("📦 Testing monitoring imports...")
        from monitoring.models import LLMTrace, QualityTrend, SafetyReport, CostAnalysis, AlertConfig
        print("✅ monitoring.models imported")
        
        from monitoring.quality import QualityMonitor
        print("✅ monitoring.quality imported")
        
        from monitoring.cost import CostTracker
        print("✅ monitoring.cost imported")
        
        # Test creating instances
        print("🏗️ Testing instance creation...")
        quality_monitor = QualityMonitor()
        print("✅ QualityMonitor instance created")
        
        cost_tracker = CostTracker()
        print("✅ CostTracker instance created")
        
        # Now test the full API server import
        print("🚀 Testing API server import...")
        from api.server import app
        print('✅ API server imports successful')
        
    except ImportError as e:
        print(f'❌ Import error: {e}')
        print("📊 Traceback:")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f'❌ Unexpected error: {e}')
        print("📊 Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 