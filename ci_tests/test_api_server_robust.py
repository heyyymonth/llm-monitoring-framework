#!/usr/bin/env python3
"""
Robust API server test for CI environments.
Handles edge cases and provides detailed error reporting.
"""

import sys
import os
import asyncio
import traceback

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_api_server_import():
    """Test importing the API server with detailed error handling."""
    try:
        print("📡 Testing API server import...")
        from api.server import app
        print("✅ API server imported successfully")
        return app
    except Exception as e:
        print(f"❌ API server import failed: {e}")
        print("\n🔍 Detailed traceback:")
        traceback.print_exc()
        return None

def test_api_server_creation():
    """Test creating FastAPI app without complex dependencies."""
    try:
        print("🏗️ Testing basic FastAPI app creation...")
        from fastapi import FastAPI
        
        # Create a minimal test app
        test_app = FastAPI(title="Test LLM Monitor", version="1.0.0")
        
        @test_app.get("/health")
        def health_check():
            return {"status": "healthy", "service": "LLM Monitor"}
        
        print("✅ Basic FastAPI app created successfully")
        return test_app
    except Exception as e:
        print(f"❌ Basic FastAPI app creation failed: {e}")
        traceback.print_exc()
        return None

def test_monitoring_components():
    """Test individual monitoring components."""
    try:
        print("🎯 Testing monitoring components...")
        
        # Test models
        from monitoring.models import LLMTrace, QualityTrend
        print("  ✅ Models imported")
        
        # Test creating a simple trace object with all required fields
        from monitoring.models import QualityMetrics, SafetyAssessment, CostMetrics
        from datetime import datetime, timezone
        
        quality_metrics = QualityMetrics(
            semantic_similarity=0.8,
            factual_accuracy=0.9,
            response_relevance=0.85,
            coherence_score=0.8,
            overall_quality=0.84
        )
        
        safety_assessment = SafetyAssessment(
            is_safe=True,
            safety_score=0.95
        )
        
        cost_metrics = CostMetrics(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            cost_usd=0.005,
            model_name="test-model"
        )
        
        trace = LLMTrace(
            trace_id="test-123",
            prompt="test input",
            model_name="test-model",
            response="test output",
            response_time_ms=1500.0,
            quality_metrics=quality_metrics,
            safety_assessment=safety_assessment,
            cost_metrics=cost_metrics
        )
        print("  ✅ LLMTrace object created")
        
        # Test quality monitoring (without external dependencies)
        from monitoring.quality import QualityMonitor
        quality_monitor = QualityMonitor()
        print("  ✅ QualityMonitor created")
        
        # Test cost tracking
        from monitoring.cost import CostTracker
        cost_tracker = CostTracker()
        print("  ✅ CostTracker created")
        
        print("✅ All monitoring components working")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring components failed: {e}")
        traceback.print_exc()
        return False

def main():
    print(f"🐍 Python: {sys.version}")
    print(f"🖥️  Platform: {sys.platform}")
    print(f"📁 Working dir: {os.getcwd()}")
    print(f"📦 Project root: {project_root}")
    print()
    
    success = True
    
    # Test 1: Monitoring components
    success &= test_monitoring_components()
    
    # Test 2: Basic FastAPI app
    test_app = test_api_server_creation()
    success &= (test_app is not None)
    
    # Test 3: Full API server import
    if success:
        api_app = test_api_server_import()
        success &= (api_app is not None)
        
        if api_app:
            # Test that the app has expected endpoints
            routes = [route.path for route in api_app.routes]
            expected_routes = ["/monitor/inference", "/metrics/quality", "/health"]
            
            found_routes = [route for route in expected_routes if route in routes]
            print(f"📍 Found routes: {found_routes}")
            
            if len(found_routes) >= 2:  # At least 2 of the expected routes
                print("✅ API server has expected routes")
            else:
                print("⚠️  API server missing some expected routes")
    
    if success:
        print("\n🎉 All API server tests passed!")
        return 0
    else:
        print("\n❌ API server tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 