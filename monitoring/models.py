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


class DiskIOMetrics(BaseModel):
    """Disk I/O metrics critical for model loading and checkpointing."""
    model_config = ConfigDict(protected_namespaces=())
    
    device: str
    read_bytes_per_sec: float = 0.0
    write_bytes_per_sec: float = 0.0
    read_iops: float = 0.0
    write_iops: float = 0.0
    read_latency_ms: float = 0.0
    write_latency_ms: float = 0.0
    disk_utilization_percent: float = 0.0
    queue_depth: float = 0.0


class NetworkIOMetrics(BaseModel):
    """Network I/O metrics for distributed inference and model communication."""
    model_config = ConfigDict(protected_namespaces=())
    
    interface: str
    bytes_sent_per_sec: float = 0.0
    bytes_recv_per_sec: float = 0.0
    packets_sent_per_sec: float = 0.0
    packets_recv_per_sec: float = 0.0
    errors_per_sec: float = 0.0
    drops_per_sec: float = 0.0
    bandwidth_utilization_percent: float = 0.0


class MemoryFragmentationMetrics(BaseModel):
    """Memory fragmentation metrics that can impact model loading."""
    model_config = ConfigDict(protected_namespaces=())
    
    largest_free_block_mb: float = 0.0
    fragmentation_percent: float = 0.0
    swap_usage_mb: float = 0.0
    swap_pressure: bool = False
    page_faults_per_sec: float = 0.0
    memory_compaction_events: int = 0


class ProcessSchedulerMetrics(BaseModel):
    """Process scheduler metrics affecting inference latency."""
    model_config = ConfigDict(protected_namespaces=())
    
    context_switches_per_sec: float = 0.0
    run_queue_length: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    scheduler_latency_ms: float = 0.0
    cpu_steal_percent: float = 0.0  # CPU stolen by hypervisor


class LLMProcessMetrics(BaseModel):
    """LLM process-specific metrics that affect inference performance."""
    model_config = ConfigDict(protected_namespaces=())
    
    pid: int
    cpu_percent: float
    memory_rss_mb: float  # Resident memory - actual RAM used by LLM
    memory_vms_mb: float  # Virtual memory size
    memory_percent: float
    model_memory_mb: float = 0.0  # Estimated model memory usage
    inference_threads: int = 1  # Active inference threads
    open_files: int = 0  # File descriptors
    context_switches: int = 0  # Process context switches
    page_faults: int = 0  # Memory page faults
    cpu_affinity: List[int] = []  # CPU cores assigned
    nice_value: int = 0  # Process priority
    io_read_bytes: int = 0  # Cumulative I/O reads
    io_write_bytes: int = 0  # Cumulative I/O writes


class SystemMetrics(BaseModel):
    """Enhanced system metrics focused on LLM inference performance."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core system metrics affecting LLM inference
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    available_memory_gb: float  # Free memory for model loading
    memory_pressure: bool = False  # System under memory pressure
    
    # Enhanced system performance metrics
    system_load_1m: float = 0.0
    system_load_5m: float = 0.0
    system_load_15m: float = 0.0
    boot_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # GPU metrics (critical for LLM inference)
    gpu_count: int = 0
    gpu_metrics: List[Dict[str, Any]] = []
    
    # Thermal metrics (affects inference speed)
    cpu_temp_celsius: float = 0.0
    thermal_throttling: bool = False
    thermal_zones: Dict[str, float] = {}  # Multiple thermal sensors
    
    # Disk I/O metrics (critical for model loading)
    disk_io_metrics: List[DiskIOMetrics] = []
    
    # Network metrics (for distributed inference)
    network_metrics: List[NetworkIOMetrics] = []
    
    # Memory fragmentation (impacts model loading)
    memory_fragmentation: Optional[MemoryFragmentationMetrics] = None
    
    # Process scheduler efficiency
    scheduler_metrics: Optional[ProcessSchedulerMetrics] = None
    
    # LLM-specific process metrics
    llm_process_metrics: Optional[LLMProcessMetrics] = None
    
    # Container metrics (if running in containers)
    container_memory_limit_gb: Optional[float] = None
    container_cpu_limit: Optional[float] = None
    container_throttled_time_ms: float = 0.0


class GPUMetrics(BaseModel):
    """GPU metrics focused on LLM inference performance."""
    model_config = ConfigDict(protected_namespaces=())
    
    gpu_id: int
    name: str
    temperature: float
    utilization_percent: float  # GPU compute utilization
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    power_draw_watts: float
    power_limit_watts: float


class InferenceMetrics(BaseModel):
    """Comprehensive LLM inference performance metrics."""
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
    
    # LLM-specific performance metrics
    memory_peak_mb: float = 0.0  # Peak memory during this inference
    gpu_utilization_percent: float = 0.0  # GPU usage during inference
    cache_hit: bool = False  # KV cache hit
    batch_size: int = 1
    sequence_length: int = 0
    
    metadata: Dict[str, Any] = {}


class ErrorMetrics(BaseModel):
    """LLM inference error metrics with context."""
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
    
    # LLM-specific error context
    memory_usage_at_error_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    tokens_processed_before_error: int = 0


class QueueMetrics(BaseModel):
    """LLM inference queue metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pending_requests: int
    processing_requests: int
    completed_requests: int
    failed_requests: int
    avg_wait_time_ms: float
    max_wait_time_ms: float
    queue_throughput: float  # requests per second


class PerformanceSummary(BaseModel):
    """LLM performance summary with inference-specific metrics."""
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
    
    # LLM performance specific
    avg_memory_usage_mb: float
    peak_memory_usage_mb: float
    avg_gpu_utilization: float
    cache_hit_rate: float
    avg_queue_time_ms: float
    thermal_throttling_events: int
    memory_pressure_events: int


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
    services: Dict[str, bool] = {}


class ModelInfo(BaseModel):
    """LLM model information and performance characteristics."""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    version: Optional[str] = None
    type: Optional[str] = None
    parameters: Optional[int] = None
    context_length: Optional[int] = None
    loaded_timestamp: Optional[datetime] = None
    memory_usage_gb: Optional[float] = None
    gpu_layers: Optional[int] = None  # Number of layers on GPU
    quantization: Optional[str] = None  # Quantization type (fp16, int8, etc.)
    metadata: Dict[str, Any] = {}


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
    """Per-model performance tracking."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_requests: int
    avg_response_time_ms: float
    error_rate: float
    memory_usage_gb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0 