"""
LLM Quality & Safety Monitoring Package

Provides comprehensive monitoring capabilities for LLM applications focusing on:
- Quality assessment and drift detection
- Safety guardrails and violation monitoring
- Cost tracking and optimization
- Real-time observability for LLM-specific metrics
"""

from .models import (
    SafetyFlag,
    QualityMetrics,
    SafetyAssessment,
    CostMetrics,
    LLMTrace,
    QualityTrend,
    SafetyReport,
    CostAnalysis,
    ModelPerformance,
    AlertConfig
)

from .quality import (
    QualityMonitor,
    HallucinationDetector,
    SafetyEvaluator,
    QualityAssessor
)

from .cost import (
    CostTracker
)

__version__ = "2.0.0"
__all__ = [
    # Models
    "SafetyFlag",
    "QualityMetrics", 
    "SafetyAssessment",
    "CostMetrics",
    "LLMTrace",
    "QualityTrend",
    "SafetyReport",
    "CostAnalysis",
    "ModelPerformance",
    "AlertConfig",
    
    # Quality Monitoring
    "QualityMonitor",
    "HallucinationDetector", 
    "SafetyEvaluator",
    "QualityAssessor",
    
    # Cost Tracking
    "CostTracker"
] 