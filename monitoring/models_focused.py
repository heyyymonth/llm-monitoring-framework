"""
Focused models for on-premise LLM performance monitoring.

Only includes metrics that directly impact local model inference performance:
- Model loading and memory allocation
- LLM process performance
- GPU utilization 
- Memory pressure and fragmentation
- Inference performance
- Thermal throttling

Excludes:
- Network monitoring (irrelevant for local models)
- Container metrics (most on-prem deployments aren't containerized)
- Generic system metrics not related to LLM performance
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelLoadingMetrics(BaseModel):
    """Metrics specifically for model loading and caching performance."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_file_path: str = ""
    model_size_gb: float = 0.0
    loading_time_ms: float = 0.0
    disk_read_mb_per_sec: float = 0.0  # During model loading
    memory_allocation_time_ms: float = 0.0
    cache_hit: bool = False


class LLMProcessMetrics(BaseModel):
    """LLM process-specific metrics that directly affect inference performance."""
    model_config = ConfigDict(protected_namespaces=())
    
    pid: int
    cpu_percent: float
    memory_rss_mb: float  # Actual RAM used by LLM process
    model_memory_mb: float = 0.0  # Memory used for model weights
    inference_threads: int = 1  # Active inference threads
    context_switches: int = 0  # Process context switches (latency impact)
    cpu_affinity: List[int] = []  # CPU cores assigned to LLM process


class MemoryPressureMetrics(BaseModel):
    """Memory pressure metrics critical for LLM performance."""
    model_config = ConfigDict(protected_namespaces=())
    
    available_memory_gb: float  # Free memory for model loading
    memory_pressure: bool = False  # System under memory pressure
    swap_usage_gb: float = 0.0  # Swap usage (very bad for LLM performance)
    largest_free_memory_block_gb: float = 0.0  # For large model loading
    memory_fragmentation_percent: float = 0.0


class SystemMetrics(BaseModel):
    """Focused system metrics for on-premise LLM performance monitoring."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Essential system metrics
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    
    # Memory pressure (critical for LLM)
    memory_pressure_metrics: Optional[MemoryPressureMetrics] = None
    
    # Thermal throttling (affects inference speed)
    cpu_temp_celsius: float = 0.0
    thermal_throttling: bool = False
    
    # GPU metrics (critical for LLM inference)
    gpu_count: int = 0
    gpu_metrics: List[Dict[str, Any]] = []
    
    # LLM-specific metrics
    llm_process_metrics: Optional[LLMProcessMetrics] = None
    model_loading_metrics: Optional[ModelLoadingMetrics] = None
    
    # Basic system load
    load_average: List[float] = [0.0, 0.0, 0.0]


class GPUMetrics(BaseModel):
    """GPU metrics for LLM inference monitoring."""
    model_config = ConfigDict(protected_namespaces=())
    
    gpu_id: int
    name: str
    temperature: float
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0


class InferenceMetrics(BaseModel):
    """LLM inference performance metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    request_id: str
    model_name: str = ""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_ms: float
    tokens_per_second: float = 0.0
    queue_time_ms: float = 0.0
    model_loading_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cache_hit: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorMetrics(BaseModel):
    """LLM error tracking metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    error_type: str
    error_message: str
    model_name: str = ""
    request_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = {}


class QueueMetrics(BaseModel):
    """LLM request queue metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    pending_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    queue_throughput: float = 0.0


class PerformanceSummary(BaseModel):
    """LLM performance summary for reporting."""
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
    avg_queue_time_ms: float = 0.0
    thermal_throttling_events: int = 0
    memory_pressure_events: int = 0


class HealthStatus(BaseModel):
    """Health status for on-premise LLM monitoring."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    last_inference: Optional[datetime] = None
    queue_size: int = 0
    error_rate_1h: float = 0.0
    avg_response_time_1h: float = 0.0
    system_metrics: Optional[SystemMetrics] = None
    llm_process_health: bool = True
    memory_pressure_detected: bool = False
    thermal_throttling_detected: bool = False 