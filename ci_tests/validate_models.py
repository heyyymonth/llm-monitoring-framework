#!/usr/bin/env python3
"""
Model validation script for CI testing.
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
print(f"📦 Project root: {project_root}")

def safe_import_models():
    """Safely import monitoring models with error handling."""
    try:
        from monitoring.models import (
            QualityMetrics, SafetyAssessment, CostMetrics, LLMTrace,
            SafetyFlag, QualityTrend, SafetyReport, CostAnalysis
        )
        from datetime import datetime
        print("✅ Model imports successful")
        return {
            'QualityMetrics': QualityMetrics,
            'SafetyAssessment': SafetyAssessment,
            'CostMetrics': CostMetrics,
            'LLMTrace': LLMTrace,
            'SafetyFlag': SafetyFlag,
            'QualityTrend': QualityTrend,
            'SafetyReport': SafetyReport,
            'CostAnalysis': CostAnalysis,
            'datetime': datetime
        }
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        traceback.print_exc()
        return None

def validate_models():
    """Validate all monitoring models."""
    models = safe_import_models()
    if not models:
        return False
    
    try:
        # Test quality metrics
        print("🎯 Testing QualityMetrics...")
        quality = models['QualityMetrics'](
            semantic_similarity=0.95,
            factual_accuracy=0.90,
            response_relevance=0.88,
            coherence_score=0.92,
            overall_quality=0.91
        )
        print("  ✅ QualityMetrics created successfully")

        # Test safety assessment
        print("🛡️  Testing SafetyAssessment...")
        safety = models['SafetyAssessment'](
            is_safe=True,
            safety_score=0.95,
            flags=[],
            details={}
        )
        print("  ✅ SafetyAssessment created successfully")

        # Test cost metrics
        print("💰 Testing CostMetrics...")
        cost = models['CostMetrics'](
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.003,
            model_name='gpt-4'
        )
        print("  ✅ CostMetrics created successfully")

        # Test complete trace
        print("📊 Testing LLMTrace...")
        trace = models['LLMTrace'](
            trace_id='test-123',
            prompt='What is AI?',
            model_name='gpt-4',
            response='AI is artificial intelligence...',
            response_time_ms=450.0,
            quality_metrics=quality,
            safety_assessment=safety,
            cost_metrics=cost
        )
        print("  ✅ LLMTrace created successfully")

        print('\n🎉 All LLM monitoring models validate successfully!')
        print(f'   Quality: {quality.overall_quality:.2f} overall score')
        print(f'   Safety: {safety.safety_score:.2f} safety score, {len(safety.flags)} flags')
        print(f'   Cost: ${cost.cost_usd:.4f} for {cost.total_tokens} tokens')
        print(f'   Trace: {trace.trace_id} - {trace.response_time_ms}ms response')
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("🔍 Starting model validation...\n")
    
    success = validate_models()
    
    if success:
        print("\n✅ Model validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Model validation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 