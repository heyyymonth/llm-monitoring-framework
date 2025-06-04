#!/usr/bin/env python3
"""
Quick functionality test for the LLM Performance Monitoring Framework.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from monitoring.models import InferenceMetrics, SystemMetrics, ErrorMetrics, AlertRule, AlertLevel, MetricType
    from monitoring.config import MonitoringConfig
    from monitoring.client import LLMMonitor
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic framework functionality."""
    print("🧪 Quick Functionality Test")
    print("=" * 40)
    
    try:
        # Test data models
        print("📊 Testing data models...")
        
        # Test InferenceMetrics
        inference_metrics = InferenceMetrics(
            request_id="test-123",
            model_name="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            response_time_ms=1500.0
        )
        assert inference_metrics.request_id == "test-123"
        assert inference_metrics.success is True
        print("✅ InferenceMetrics working")
        
        # Test SystemMetrics
        system_metrics = SystemMetrics(
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_gb=8.1,
            memory_total_gb=16.0,
            disk_percent=30.0
        )
        assert system_metrics.cpu_percent == 45.5
        print("✅ SystemMetrics working")
        
        # Test ErrorMetrics
        error_metrics = ErrorMetrics(
            request_id="error-123",
            error_type="ValueError",
            error_message="Test error",
            model_name="test-model"
        )
        assert error_metrics.error_type == "ValueError"
        print("✅ ErrorMetrics working")
        
        # Test AlertRule
        alert_rule = AlertRule(
            id="test_rule",
            name="Test Rule",
            metric_type=MetricType.SYSTEM,
            metric_name="cpu_percent",
            threshold=80.0,
            comparison="gt",
            severity=AlertLevel.WARNING
        )
        assert alert_rule.threshold == 80.0
        print("✅ AlertRule working")
        
        # Test configuration
        print("⚙️  Testing configuration...")
        config = MonitoringConfig()
        assert config.metrics_interval > 0
        print("✅ Configuration working")
        
        # Test client (mock for quick test)
        print("🔌 Testing client...")
        monitor = LLMMonitor("http://localhost:8000")
        assert monitor.monitor_url == "http://localhost:8000"
        print("✅ Client working")
        
        # Test data serialization
        print("💾 Testing data serialization...")
        inference_json = inference_metrics.model_dump_json()
        assert len(inference_json) > 0
        print("✅ JSON serialization working")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_basic_functionality()
    
    if success:
        print("\n🌟 Framework is ready to use!")
        print("📖 Next steps:")
        print("   1. Start services: python main.py")
        print("   2. Open dashboard: http://localhost:8080")
        print("   3. Check API docs: http://localhost:8000/docs")
        print("   4. Run full tests: python tests/run_tests.py")
        print("   5. Test with real LLM: python examples/integrations/test_ollama_llm.py")
    else:
        print("\n❌ Framework setup has issues")
        print("💡 Try:")
        print("   1. pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Run from project root directory")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 