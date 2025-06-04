import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from collections import deque, defaultdict
import logging

from .models import (
    SystemMetrics, GPUMetrics, InferenceMetrics, ErrorMetrics,
    QueueMetrics, TokenUsageMetrics, PerformanceSummary
)
from .config import get_config

# Try to import GPU monitoring
try:
    import pynvml
    # Only initialize if we can import successfully
    try:
        pynvml.nvmlInit()
        GPU_AVAILABLE = True
    except Exception:
        # GPU libraries available but can't initialize (no GPU, no drivers, etc.)
        GPU_AVAILABLE = False
        pynvml = None
except ImportError:
    # GPU libraries not available
    GPU_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        self.config = get_config()
        self._running = False
        self._metrics_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # In-memory storage for recent metrics
        self._system_metrics: deque = deque(maxlen=1000)
        self._inference_metrics: deque = deque(maxlen=10000)
        self._error_metrics: deque = deque(maxlen=1000)
        self._queue_metrics: deque = deque(maxlen=1000)
        
        # Performance tracking
        self._request_times: defaultdict = defaultdict(list)
        self._token_counts: defaultdict = defaultdict(list)
        self._error_counts: defaultdict = defaultdict(int)
        
        # Queue tracking
        self._pending_requests = 0
        self._processing_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._wait_times: deque = deque(maxlen=1000)
        
        self.start_time = datetime.utcnow()
        
    def start(self):
        """Start the metrics collection background thread."""
        if not self._running:
            self._running = True
            self._metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self._metrics_thread.start()
            logger.info("Metrics collector started")
    
    def stop(self):
        """Stop the metrics collection."""
        self._running = False
        if self._metrics_thread:
            self._metrics_thread.join()
        logger.info("Metrics collector stopped")
    
    def _collect_metrics_loop(self):
        """Background loop for collecting system metrics."""
        while self._running:
            try:
                system_metrics = self._collect_system_metrics()
                with self._lock:
                    self._system_metrics.append(system_metrics)
                
                queue_metrics = self._collect_queue_metrics()
                with self._lock:
                    self._queue_metrics.append(queue_metrics)
                    
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.config.monitoring.metrics_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # Load average (Unix systems)
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            load_avg = []
        
        # GPU metrics
        gpu_metrics = []
        gpu_count = 0
        
        if GPU_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Basic info
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Power
                    try:
                        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                    except pynvml.NVMLError:
                        power_draw = 0.0
                        power_limit = 0.0
                    
                    gpu_metric = GPUMetrics(
                        gpu_id=i,
                        name=name,
                        temperature=temp,
                        utilization_percent=util.gpu,
                        memory_used_mb=mem_info.used / 1024 / 1024,
                        memory_total_mb=mem_info.total / 1024 / 1024,
                        memory_percent=(mem_info.used / mem_info.total) * 100,
                        power_draw_watts=power_draw,
                        power_limit_watts=power_limit
                    )
                    gpu_metrics.append(gpu_metric.model_dump())
                    
            except Exception as e:
                logger.warning(f"Error collecting GPU metrics: {e}")
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            gpu_count=gpu_count,
            gpu_metrics=gpu_metrics,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            load_average=load_avg
        )
    
    def _collect_queue_metrics(self) -> QueueMetrics:
        """Collect current queue metrics."""
        with self._lock:
            avg_wait_time = sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0.0
            max_wait_time = max(self._wait_times) if self._wait_times else 0.0
            
            # Calculate throughput (requests per second over last minute)
            now = datetime.utcnow()
            recent_completions = sum(1 for t in self._wait_times if (now - datetime.fromtimestamp(t/1000)).seconds < 60)
            throughput = recent_completions / 60.0
        
        return QueueMetrics(
            pending_requests=self._pending_requests,
            processing_requests=self._processing_requests,
            completed_requests=self._completed_requests,
            failed_requests=self._failed_requests,
            avg_wait_time_ms=avg_wait_time,
            max_wait_time_ms=max_wait_time,
            queue_throughput=throughput
        )
    
    def log_inference(self, metrics: InferenceMetrics):
        """Log inference metrics."""
        with self._lock:
            self._inference_metrics.append(metrics)
            
            # Update performance tracking
            minute_key = metrics.timestamp.replace(second=0, microsecond=0)
            self._request_times[minute_key].append(metrics.response_time_ms)
            self._token_counts[minute_key].append(metrics.total_tokens)
            
            if not metrics.success:
                self._error_counts[minute_key] += 1
                self._failed_requests += 1
            else:
                self._completed_requests += 1
    
    def log_error(self, metrics: ErrorMetrics):
        """Log error metrics."""
        with self._lock:
            self._error_metrics.append(metrics)
            self._failed_requests += 1
    
    def increment_queue_pending(self):
        """Increment pending request count."""
        with self._lock:
            self._pending_requests += 1
    
    def decrement_queue_pending(self):
        """Decrement pending request count."""
        with self._lock:
            self._pending_requests = max(0, self._pending_requests - 1)
    
    def increment_queue_processing(self):
        """Increment processing request count."""
        with self._lock:
            self._processing_requests += 1
    
    def decrement_queue_processing(self):
        """Decrement processing request count."""
        with self._lock:
            self._processing_requests = max(0, self._processing_requests - 1)
    
    def log_wait_time(self, wait_time_ms: float):
        """Log a request wait time."""
        with self._lock:
            self._wait_times.append(wait_time_ms)
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        with self._lock:
            return self._system_metrics[-1] if self._system_metrics else None
    
    def get_recent_inference_metrics(self, limit: int = 100) -> List[InferenceMetrics]:
        """Get recent inference metrics."""
        with self._lock:
            return list(self._inference_metrics)[-limit:]
    
    def get_recent_error_metrics(self, limit: int = 100) -> List[ErrorMetrics]:
        """Get recent error metrics."""
        with self._lock:
            return list(self._error_metrics)[-limit:]
    
    def get_performance_summary(self, time_period: str = "1h") -> PerformanceSummary:
        """Get performance summary for a time period."""
        
        # Parse time period
        if time_period == "1h":
            cutoff = datetime.utcnow() - timedelta(hours=1)
        elif time_period == "24h":
            cutoff = datetime.utcnow() - timedelta(days=1)
        elif time_period == "7d":
            cutoff = datetime.utcnow() - timedelta(days=7)
        else:
            cutoff = datetime.utcnow() - timedelta(hours=1)
        
        with self._lock:
            # Filter metrics by time period
            recent_inferences = [m for m in self._inference_metrics if m.timestamp >= cutoff]
            recent_system = [m for m in self._system_metrics if m.timestamp >= cutoff]
        
        if not recent_inferences:
            return PerformanceSummary(
                time_period=time_period,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=0.0,
                avg_response_time_ms=0.0,
                median_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                avg_tokens_per_second=0.0,
                total_tokens_processed=0,
                avg_cpu_percent=0.0,
                avg_memory_percent=0.0,
                avg_gpu_utilization=0.0,
                peak_memory_usage_gb=0.0
            )
        
        # Calculate inference metrics
        successful = [m for m in recent_inferences if m.success]
        failed = [m for m in recent_inferences if not m.success]
        
        response_times = [m.response_time_ms for m in successful]
        response_times.sort()
        
        total_tokens = sum(m.total_tokens for m in recent_inferences)
        avg_tokens_per_second = sum(m.tokens_per_second for m in successful) / len(successful) if successful else 0.0
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            return data[f] + (data[f + 1] - data[f]) * c if f + 1 < len(data) else data[f]
        
        # Calculate system metrics
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system) if recent_system else 0.0
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system) if recent_system else 0.0
        peak_memory = max((m.memory_used_gb for m in recent_system), default=0.0)
        
        # Calculate average GPU utilization
        avg_gpu = 0.0
        gpu_count = 0
        for m in recent_system:
            if m.gpu_metrics:
                gpu_utils = [gpu['utilization_percent'] for gpu in m.gpu_metrics]
                avg_gpu += sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
                gpu_count += 1
        
        if gpu_count > 0:
            avg_gpu /= gpu_count
        
        return PerformanceSummary(
            time_period=time_period,
            total_requests=len(recent_inferences),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=(len(failed) / len(recent_inferences)) * 100 if recent_inferences else 0.0,
            avg_response_time_ms=sum(response_times) / len(response_times) if response_times else 0.0,
            median_response_time_ms=percentile(response_times, 0.5),
            p95_response_time_ms=percentile(response_times, 0.95),
            p99_response_time_ms=percentile(response_times, 0.99),
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens_processed=total_tokens,
            avg_cpu_percent=avg_cpu,
            avg_memory_percent=avg_memory,
            avg_gpu_utilization=avg_gpu,
            peak_memory_usage_gb=peak_memory
        )
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def cleanup_old_metrics(self):
        """Clean up old metrics based on configuration."""
        cutoff = datetime.utcnow() - timedelta(days=self.config.monitoring.max_history_days)
        
        with self._lock:
            # Filter out old metrics
            self._system_metrics = deque(
                (m for m in self._system_metrics if m.timestamp >= cutoff),
                maxlen=self._system_metrics.maxlen
            )
            self._inference_metrics = deque(
                (m for m in self._inference_metrics if m.timestamp >= cutoff),
                maxlen=self._inference_metrics.maxlen
            )
            self._error_metrics = deque(
                (m for m in self._error_metrics if m.timestamp >= cutoff),
                maxlen=self._error_metrics.maxlen
            ) 