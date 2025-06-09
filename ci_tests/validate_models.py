#!/usr/bin/env python3
"""
Model validation script for CI testing.
"""

from monitoring.models import (
    QualityMetrics, SafetyAssessment, CostMetrics, LLMTrace,
    SafetyFlag, QualityTrend, SafetyReport, CostAnalysis
)
from datetime import datetime

def main():
    # Test quality metrics
    quality = QualityMetrics(
        semantic_similarity=0.95,
        factual_accuracy=0.90,
        response_relevance=0.88,
        coherence_score=0.92,
        overall_quality=0.91
    )

    # Test safety assessment
    safety = SafetyAssessment(
        is_safe=True,
        safety_score=0.95,
        flags=[],
        details={}
    )

    # Test cost metrics
    cost = CostMetrics(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost_usd=0.003,
        model_name='gpt-4'
    )

    # Test complete trace
    trace = LLMTrace(
        trace_id='test-123',
        prompt='What is AI?',
        model_name='gpt-4',
        response='AI is artificial intelligence...',
        response_time_ms=450.0,
        quality_metrics=quality,
        safety_assessment=safety,
        cost_metrics=cost
    )

    print('✅ All LLM monitoring models validate successfully')
    print(f'   Quality: {quality.overall_quality:.2f} overall score')
    print(f'   Safety: {safety.safety_score:.2f} safety score, {len(safety.flags)} flags')
    print(f'   Cost: ${cost.cost_usd:.4f} for {cost.total_tokens} tokens')
    print(f'   Trace: {trace.trace_id} - {trace.response_time_ms}ms response')

if __name__ == '__main__':
    main() 