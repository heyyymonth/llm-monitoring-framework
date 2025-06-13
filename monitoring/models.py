from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class SafetyFlag(str, Enum):
    """Safety flags for LLM outputs."""
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity" 
    BIAS = "bias"
    PII_LEAK = "pii_leak"
    PROMPT_INJECTION = "prompt_injection"


class QualityMetrics(BaseModel):
    """Quality assessment metrics for LLM responses."""
    model_config = ConfigDict(protected_namespaces=())
    
    # Core quality scores (0-1)
    semantic_similarity: float = Field(ge=0, le=1)
    factual_accuracy: float = Field(ge=0, le=1) 
    response_relevance: float = Field(ge=0, le=1)
    coherence_score: float = Field(ge=0, le=1)
    response_length: int = Field(ge=0)
    
    # Overall quality (weighted average)
    overall_quality: float = Field(ge=0, le=1)


class SafetyAssessment(BaseModel):
    """Safety evaluation results."""
    model_config = ConfigDict(protected_namespaces=())
    
    is_safe: bool
    safety_score: float = Field(ge=0, le=1)
    flags: List[SafetyFlag] = []
    details: Dict[str, Any] = {}


class CostMetrics(BaseModel):
    """Cost tracking for LLM usage."""
    model_config = ConfigDict(protected_namespaces=())
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    model_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMTrace(BaseModel):
    """Complete trace of an LLM interaction."""
    model_config = ConfigDict(protected_namespaces=())
    
    trace_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Request data
    prompt: str
    model_name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Response data  
    response: str
    response_time_ms: float
    
    # Quality assessment
    quality_metrics: QualityMetrics
    safety_assessment: SafetyAssessment
    cost_metrics: CostMetrics
    
    # User feedback
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    user_feedback: Optional[str] = None


class QualityTrend(BaseModel):
    """Quality trends over time."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    average_quality: float = Field(ge=0, le=1)
    quality_decline: bool
    decline_percentage: Optional[float] = None
    top_issues: List[str] = []


class SafetyReport(BaseModel):
    """Safety incident reporting."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    total_interactions: int
    safety_violations: int
    violation_rate: float = Field(ge=0, le=1)
    common_flags: List[SafetyFlag] = []
    critical_incidents: int = 0


class CostAnalysis(BaseModel):
    """Cost analysis and optimization insights."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    total_cost_usd: float
    avg_cost_per_request: float
    most_expensive_operations: List[str] = []
    optimization_suggestions: List[str] = []
    projected_monthly_cost: float


class ModelPerformance(BaseModel):
    """Model performance summary."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    time_period: str
    
    # Performance metrics
    avg_response_time_ms: float
    p95_response_time_ms: float
    throughput_requests_per_minute: float
    
    # Quality metrics
    avg_quality_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    user_satisfaction: float = Field(ge=0, le=1)
    
    # Business metrics
    task_completion_rate: float = Field(ge=0, le=1)
    user_engagement_rate: float = Field(ge=0, le=1)


class AlertConfig(BaseModel):
    """Alert configuration for monitoring."""
    model_config = ConfigDict(protected_namespaces=())
    
    quality_threshold: float = Field(0.8, ge=0, le=1)
    safety_threshold: float = Field(0.9, ge=0, le=1)
    cost_threshold_usd: float = 100.0
    response_time_threshold_ms: float = 2000.0
    
    alert_channels: List[str] = ["email", "slack"]
    escalation_enabled: bool = True 