from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class LLMProcessMetrics(BaseModel):
    """LLM process-specific metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    pid: int
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    model_memory_mb: float = 0.0
    inference_threads: int = 1
    gpu_memory_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    model_loading_time_ms: float = 0.0
    peak_memory_mb: float = 0.0


class SystemMetrics(BaseModel):
    """Essential system metrics for LLM monitoring."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    memory_total_gb: float
    system_load_1m: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mbps: float = 0.0
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
    
    # LLM-specific metrics
    memory_peak_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cache_hit: bool = False
    queue_time_ms: float = 0.0


class PerformanceSummary(BaseModel):
    """LLM performance summary."""
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    avg_tokens_per_second: float
    total_tokens_processed: int
    avg_memory_usage_mb: float
    peak_memory_usage_mb: float
    avg_gpu_utilization: float
    cache_hit_rate: float
    avg_queue_time_ms: float
    thermal_throttling_events: int
    memory_pressure_events: int


class HealthStatus(BaseModel):
    """Basic health status."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    active_connections: int = 0
    queue_size: int = 0
    error_rate_1h: float = 0.0
    avg_response_time_1h: float = 0.0 