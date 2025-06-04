from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class MetricType(str, Enum):
    SYSTEM = "system"
    INFERENCE = "inference"
    ERROR = "error"
    CUSTOM = "custom"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    gpu_count: int = 0
    gpu_metrics: List[Dict[str, Any]] = []
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    load_average: List[float] = []


class GPUMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    gpu_id: int
    name: str
    temperature: float
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    power_draw_watts: float
    power_limit_watts: float


class InferenceMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str
    model_name: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_ms: float
    queue_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_length: int = 0
    response_length: int = 0
    success: bool = True
    error_message: Optional[str] = None
    model_version: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = {}


class ErrorMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    model_name: Optional[str] = None
    endpoint: Optional[str] = None
    user_id: Optional[str] = None
    severity: AlertLevel = AlertLevel.ERROR


class PerformanceSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    time_period: str  # e.g., "1h", "24h", "7d"
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
    avg_cpu_percent: float
    avg_memory_percent: float
    avg_gpu_utilization: float
    peak_memory_usage_gb: float


class AlertRule(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    name: str
    metric_type: MetricType
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    severity: AlertLevel
    enabled: bool = True
    description: Optional[str] = None


class Alert(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rule_id: str
    rule_name: str
    severity: AlertLevel
    message: str
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class HealthStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    last_inference: Optional[datetime] = None
    active_connections: int = 0
    queue_size: int = 0
    error_rate_1h: float = 0.0
    avg_response_time_1h: float = 0.0
    system_metrics: SystemMetrics
    services: Dict[str, bool] = {}  # database, redis, etc.


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    version: Optional[str] = None
    type: Optional[str] = None
    parameters: Optional[int] = None
    context_length: Optional[int] = None
    loaded_timestamp: Optional[datetime] = None
    memory_usage_gb: Optional[float] = None
    metadata: Dict[str, Any] = {}


class QueueMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pending_requests: int
    processing_requests: int
    completed_requests: int
    failed_requests: int
    avg_wait_time_ms: float
    max_wait_time_ms: float
    queue_throughput: float  # requests per second


class TokenUsageMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    time_period: str  # e.g., "1m", "5m", "1h"
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    avg_input_length: float
    avg_output_length: float
    token_rate: float  # tokens per second
    unique_users: int = 0


class ModelPerformanceMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_requests: int
    avg_response_time_ms: float
    avg_tokens_per_second: float
    error_rate: float
    memory_usage_gb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0 