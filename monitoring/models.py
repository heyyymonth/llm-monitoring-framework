from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMProcessMetrics(BaseModel):
    """LLM process-specific metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    pid: int
    cpu_percent: float
    memory_rss_mb: float
    memory_percent: float


class SystemMetrics(BaseModel):
    """Essential system metrics for LLM monitoring."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    memory_total_gb: float
    llm_process: Optional[LLMProcessMetrics] = None


class InferenceMetrics(BaseModel):
    """LLM inference performance metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str
    model_name: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_ms: float
    tokens_per_second: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class PerformanceSummary(BaseModel):
    """LLM performance summary."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    avg_tokens_per_second: float
    total_tokens_processed: int


class HealthStatus(BaseModel):
    """Basic health status."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float 