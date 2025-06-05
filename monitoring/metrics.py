import asyncio
import time
import psutil
import threading
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from collections import deque, defaultdict
import logging

from .models import (
    SystemMetrics, GPUMetrics, InferenceMetrics, ErrorMetrics,
    QueueMetrics, TokenUsageMetrics, PerformanceSummary,
    LLMProcessMetrics, DiskIOMetrics, NetworkIOMetrics,
    MemoryFragmentationMetrics, ProcessSchedulerMetrics
)
from .config import get_config

# Try to import GPU monitoring
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False
        pynvml = None
except ImportError:
    GPU_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects LLM inference-specific metrics."""
    
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
        
        # Get current process for LLM monitoring
        self._current_process = psutil.Process()
        
        # Thermal monitoring state
        self._thermal_throttling_events = 0
        self._memory_pressure_events = 0
        
        # Enhanced metrics tracking
        self._last_disk_io = {}  # For calculating rates
        self._last_network_stats = {}  # For calculating rates
        self._last_collection_time = time.time()
        
    def start(self):
        """Start the metrics collection background thread."""
        if not self._running:
            self._running = True
            self._metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self._metrics_thread.start()
            logger.info("LLM metrics collector started")
    
    def stop(self):
        """Stop the metrics collection."""
        self._running = False
        if self._metrics_thread:
            self._metrics_thread.join()
        logger.info("LLM metrics collector stopped")
    
    def _collect_metrics_loop(self):
        """Background loop for collecting LLM-specific metrics."""
        while self._running:
            try:
                system_metrics = self._collect_llm_system_metrics()
                with self._lock:
                    self._system_metrics.append(system_metrics)
                
                queue_metrics = self._collect_queue_metrics()
                with self._lock:
                    self._queue_metrics.append(queue_metrics)
                
                # Update collection time for rate calculations
                self._last_collection_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error collecting LLM metrics: {e}")
            
            time.sleep(self.config.monitoring.metrics_interval)
    
    def _collect_llm_system_metrics(self) -> SystemMetrics:
        """Collect system metrics relevant to LLM inference performance."""
        # Core system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Calculate available memory for model loading
        available_memory_gb = memory.available / (1024**3)
        
        # Check for memory pressure (simplified heuristic)
        memory_pressure = memory.percent > 85 or memory.available < (2 * 1024**3)  # Less than 2GB free
        if memory_pressure:
            self._memory_pressure_events += 1
        
        # Thermal monitoring (simplified)
        cpu_temp = 0.0
        thermal_throttling = False
        
        try:
            # Try to get CPU temperature
            sensors = psutil.sensors_temperatures()
            if sensors:
                for name, entries in sensors.items():
                    if name.lower() in ['coretemp', 'cpu', 'cpu_thermal']:
                        for entry in entries:
                            cpu_temp = max(cpu_temp, entry.current)
                            if entry.critical and entry.current > entry.critical * 0.9:
                                thermal_throttling = True
                                self._thermal_throttling_events += 1
        except (AttributeError, Exception):
            pass  # Thermal monitoring not available
        
        # GPU metrics
        gpu_metrics = []
        gpu_count = 0
        
        if GPU_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    try:
                        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
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
        
        # Enhanced system metrics
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = time.time() - psutil.boot_time()
        
        # Enhanced thermal monitoring
        thermal_zones = self._collect_thermal_zones()
        
        # Disk I/O metrics
        disk_io_metrics = self._collect_disk_io_metrics()
        
        # Network metrics
        network_metrics = self._collect_network_metrics()
        
        # Memory fragmentation
        memory_fragmentation = self._collect_memory_fragmentation()
        
        # Process scheduler metrics
        scheduler_metrics = self._collect_scheduler_metrics()
        
        # Container metrics
        container_memory_limit, container_cpu_limit, container_throttled = self._collect_container_metrics()
        
        # LLM process metrics
        llm_process_metrics = self._collect_enhanced_llm_process_metrics()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            available_memory_gb=available_memory_gb,
            memory_pressure=memory_pressure,
            system_load_1m=load_avg[0],
            system_load_5m=load_avg[1],
            system_load_15m=load_avg[2],
            boot_time=boot_time,
            uptime_seconds=uptime,
            gpu_count=gpu_count,
            gpu_metrics=gpu_metrics,
            cpu_temp_celsius=cpu_temp,
            thermal_throttling=thermal_throttling,
            thermal_zones=thermal_zones,
            disk_io_metrics=disk_io_metrics,
            network_metrics=network_metrics,
            memory_fragmentation=memory_fragmentation,
            scheduler_metrics=scheduler_metrics,
            llm_process_metrics=llm_process_metrics,
            container_memory_limit_gb=container_memory_limit,
            container_cpu_limit=container_cpu_limit,
            container_throttled_time_ms=container_throttled
        )
    
    def _collect_llm_process_metrics(self) -> Optional[LLMProcessMetrics]:
        """Collect metrics specific to the LLM process."""
        try:
            memory_info = self._current_process.memory_info()
            
            # Estimate model memory usage (simplified heuristic)
            model_memory_mb = memory_info.rss / 1024 / 1024  # Assume most RSS is model
            
            # Count inference threads (simplified - count all threads)
            inference_threads = self._current_process.num_threads()
            
            return LLMProcessMetrics(
                pid=self._current_process.pid,
                cpu_percent=self._current_process.cpu_percent(),
                memory_rss_mb=memory_info.rss / 1024 / 1024,
                memory_percent=self._current_process.memory_percent(),
                model_memory_mb=model_memory_mb,
                inference_threads=inference_threads
            )
        except Exception as e:
            logger.warning(f"Error collecting LLM process metrics: {e}")
            return None
    
    def _collect_thermal_zones(self) -> Dict[str, float]:
        """Collect detailed thermal sensor information."""
        thermal_zones = {}
        try:
            sensors = psutil.sensors_temperatures()
            if sensors:
                for zone_name, entries in sensors.items():
                    for i, entry in enumerate(entries):
                        key = f"{zone_name}_{i}" if len(entries) > 1 else zone_name
                        thermal_zones[key] = entry.current
        except (AttributeError, Exception):
            pass
        return thermal_zones
    
    def _collect_disk_io_metrics(self) -> List[DiskIOMetrics]:
        """Collect disk I/O metrics for model loading performance analysis."""
        disk_metrics = []
        current_time = time.time()
        time_delta = current_time - self._last_collection_time
        
        try:
            disk_io = psutil.disk_io_counters(perdisk=True)
            for device, counters in disk_io.items():
                # Calculate rates if we have previous data
                if device in self._last_disk_io and time_delta > 0:
                    last_counters = self._last_disk_io[device]
                    read_bytes_per_sec = (counters.read_bytes - last_counters.read_bytes) / time_delta
                    write_bytes_per_sec = (counters.write_bytes - last_counters.write_bytes) / time_delta
                    read_iops = (counters.read_count - last_counters.read_count) / time_delta
                    write_iops = (counters.write_count - last_counters.write_count) / time_delta
                    
                    # Calculate approximate latency (simplified)
                    read_time_delta = counters.read_time - last_counters.read_time
                    write_time_delta = counters.write_time - last_counters.write_time
                    read_count_delta = counters.read_count - last_counters.read_count
                    write_count_delta = counters.write_count - last_counters.write_count
                    
                    read_latency = (read_time_delta / read_count_delta) if read_count_delta > 0 else 0.0
                    write_latency = (write_time_delta / write_count_delta) if write_count_delta > 0 else 0.0
                else:
                    read_bytes_per_sec = write_bytes_per_sec = 0.0
                    read_iops = write_iops = 0.0
                    read_latency = write_latency = 0.0
                
                # Store current counters for next calculation
                self._last_disk_io[device] = counters
                
                disk_metrics.append(DiskIOMetrics(
                    device=device,
                    read_bytes_per_sec=read_bytes_per_sec,
                    write_bytes_per_sec=write_bytes_per_sec,
                    read_iops=read_iops,
                    write_iops=write_iops,
                    read_latency_ms=read_latency,
                    write_latency_ms=write_latency,
                    disk_utilization_percent=0.0,  # TODO: Calculate from busy_time
                    queue_depth=0.0  # Not available in psutil
                ))
        except Exception as e:
            logger.warning(f"Error collecting disk I/O metrics: {e}")
        
        return disk_metrics
    
    def _collect_network_metrics(self) -> List[NetworkIOMetrics]:
        """Collect network I/O metrics for distributed inference monitoring."""
        network_metrics = []
        current_time = time.time()
        time_delta = current_time - self._last_collection_time
        
        try:
            net_io = psutil.net_io_counters(pernic=True)
            for interface, counters in net_io.items():
                # Skip loopback interface
                if interface.startswith('lo'):
                    continue
                
                # Calculate rates if we have previous data
                if interface in self._last_network_stats and time_delta > 0:
                    last_counters = self._last_network_stats[interface]
                    bytes_sent_per_sec = (counters.bytes_sent - last_counters.bytes_sent) / time_delta
                    bytes_recv_per_sec = (counters.bytes_recv - last_counters.bytes_recv) / time_delta
                    packets_sent_per_sec = (counters.packets_sent - last_counters.packets_sent) / time_delta
                    packets_recv_per_sec = (counters.packets_recv - last_counters.packets_recv) / time_delta
                    errors_per_sec = (counters.errin + counters.errout - 
                                    last_counters.errin - last_counters.errout) / time_delta
                    drops_per_sec = (counters.dropin + counters.dropout - 
                                   last_counters.dropin - last_counters.dropout) / time_delta
                else:
                    bytes_sent_per_sec = bytes_recv_per_sec = 0.0
                    packets_sent_per_sec = packets_recv_per_sec = 0.0
                    errors_per_sec = drops_per_sec = 0.0
                
                # Store current counters for next calculation
                self._last_network_stats[interface] = counters
                
                network_metrics.append(NetworkIOMetrics(
                    interface=interface,
                    bytes_sent_per_sec=bytes_sent_per_sec,
                    bytes_recv_per_sec=bytes_recv_per_sec,
                    packets_sent_per_sec=packets_sent_per_sec,
                    packets_recv_per_sec=packets_recv_per_sec,
                    errors_per_sec=errors_per_sec,
                    drops_per_sec=drops_per_sec,
                    bandwidth_utilization_percent=0.0  # Would need interface speed info
                ))
        except Exception as e:
            logger.warning(f"Error collecting network metrics: {e}")
        
        return network_metrics
    
    def _collect_memory_fragmentation(self) -> Optional[MemoryFragmentationMetrics]:
        """Collect memory fragmentation metrics that can impact model loading."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Calculate fragmentation (simplified heuristic)
            # This is a rough estimate - real fragmentation requires kernel-level info
            fragmentation_percent = max(0, 100 - (memory.available / memory.total * 100))
            
            # Detect swap pressure
            swap_pressure = swap.percent > 25  # More than 25% swap usage indicates pressure
            
            return MemoryFragmentationMetrics(
                largest_free_block_mb=memory.available / (1024 * 1024),
                fragmentation_percent=fragmentation_percent,
                swap_usage_mb=swap.used / (1024 * 1024),
                swap_pressure=swap_pressure,
                page_faults_per_sec=0.0,  # Would need to track over time
                memory_compaction_events=0  # Not available in psutil
            )
        except Exception as e:
            logger.warning(f"Error collecting memory fragmentation metrics: {e}")
            return None
    
    def _collect_scheduler_metrics(self) -> Optional[ProcessSchedulerMetrics]:
        """Collect process scheduler metrics affecting inference latency."""
        try:
            # Get load averages
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
            
            # Calculate context switches per second (simplified)
            cpu_stats = psutil.cpu_stats() if hasattr(psutil, 'cpu_stats') else None
            context_switches_per_sec = 0.0
            if cpu_stats:
                # This would need to be tracked over time for accurate rate
                context_switches_per_sec = cpu_stats.ctx_switches / self.get_uptime()
            
            return ProcessSchedulerMetrics(
                context_switches_per_sec=context_switches_per_sec,
                run_queue_length=0.0,  # Not available in psutil
                load_average_1m=load_avg[0],
                load_average_5m=load_avg[1],
                load_average_15m=load_avg[2],
                scheduler_latency_ms=0.0,  # Would need specialized tools
                cpu_steal_percent=0.0  # Would need virtualization info
            )
        except Exception as e:
            logger.warning(f"Error collecting scheduler metrics: {e}")
            return None
    
    def _collect_container_metrics(self):
        """Collect container metrics if running in a containerized environment."""
        try:
            # Check for cgroup files (Docker/Kubernetes)
            memory_limit_gb = None
            cpu_limit = None
            throttled_time_ms = 0.0
            
            # Try to read cgroup v1 memory limit
            try:
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    limit_bytes = int(f.read().strip())
                    # Check if it's not the default unlimited value
                    if limit_bytes < (1 << 62):  # Not unlimited
                        memory_limit_gb = limit_bytes / (1024**3)
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            
            # Try to read CPU limits
            try:
                with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                    quota = int(f.read().strip())
                with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                    period = int(f.read().strip())
                if quota > 0 and period > 0:
                    cpu_limit = quota / period
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            
            # Try to read throttling info
            try:
                with open('/sys/fs/cgroup/cpu/cpu.stat', 'r') as f:
                    for line in f:
                        if line.startswith('throttled_time'):
                            throttled_time_ms = int(line.split()[1]) / 1000000  # ns to ms
                            break
            except (FileNotFoundError, PermissionError, ValueError):
                pass
            
            return memory_limit_gb, cpu_limit, throttled_time_ms
        except Exception as e:
            logger.warning(f"Error collecting container metrics: {e}")
            return None, None, 0.0
    
    def _collect_enhanced_llm_process_metrics(self) -> Optional[LLMProcessMetrics]:
        """Collect enhanced metrics specific to the LLM process."""
        try:
            memory_info = self._current_process.memory_info()
            
            # Get additional process information
            try:
                open_files = self._current_process.num_fds() if hasattr(self._current_process, 'num_fds') else 0
            except (psutil.AccessDenied, AttributeError):
                open_files = 0
            
            try:
                num_ctx_switches = self._current_process.num_ctx_switches()
                context_switches = num_ctx_switches.voluntary + num_ctx_switches.involuntary
            except (psutil.AccessDenied, AttributeError):
                context_switches = 0
            
            try:
                cpu_affinity = self._current_process.cpu_affinity() if hasattr(self._current_process, 'cpu_affinity') else []
            except (psutil.AccessDenied, AttributeError):
                cpu_affinity = []
            
            try:
                nice_value = self._current_process.nice()
            except (psutil.AccessDenied, AttributeError):
                nice_value = 0
            
            try:
                io_counters = self._current_process.io_counters()
                io_read_bytes = io_counters.read_bytes
                io_write_bytes = io_counters.write_bytes
            except (psutil.AccessDenied, AttributeError):
                io_read_bytes = io_write_bytes = 0
            
            # Estimate model memory usage (simplified heuristic)
            model_memory_mb = memory_info.rss / 1024 / 1024  # Assume most RSS is model
            
            # Count inference threads (simplified - count all threads)
            inference_threads = self._current_process.num_threads()
            
            return LLMProcessMetrics(
                pid=self._current_process.pid,
                cpu_percent=self._current_process.cpu_percent(),
                memory_rss_mb=memory_info.rss / 1024 / 1024,
                memory_vms_mb=memory_info.vms / 1024 / 1024,
                memory_percent=self._current_process.memory_percent(),
                model_memory_mb=model_memory_mb,
                inference_threads=inference_threads,
                open_files=open_files,
                context_switches=context_switches,
                page_faults=0,  # Would need to track over time
                cpu_affinity=cpu_affinity,
                nice_value=nice_value,
                io_read_bytes=io_read_bytes,
                io_write_bytes=io_write_bytes
            )
        except Exception as e:
            logger.warning(f"Error collecting enhanced LLM process metrics: {e}")
            return None

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
                avg_memory_usage_mb=0.0,
                peak_memory_usage_mb=0.0,
                avg_gpu_utilization=0.0,
                cache_hit_rate=0.0,
                avg_queue_time_ms=0.0,
                thermal_throttling_events=0,
                memory_pressure_events=0
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
        
        # Calculate LLM-specific metrics
        avg_memory_usage = sum(m.memory_peak_mb for m in recent_inferences) / len(recent_inferences) if recent_inferences else 0.0
        peak_memory_usage = max((m.memory_peak_mb for m in recent_inferences), default=0.0)
        
        avg_gpu = sum(m.gpu_utilization_percent for m in recent_inferences) / len(recent_inferences) if recent_inferences else 0.0
        cache_hits = sum(1 for m in recent_inferences if m.cache_hit)
        cache_hit_rate = (cache_hits / len(recent_inferences)) * 100 if recent_inferences else 0.0
        
        avg_queue_time = sum(m.queue_time_ms for m in recent_inferences) / len(recent_inferences) if recent_inferences else 0.0
        
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
            avg_memory_usage_mb=avg_memory_usage,
            peak_memory_usage_mb=peak_memory_usage,
            avg_gpu_utilization=avg_gpu,
            cache_hit_rate=cache_hit_rate,
            avg_queue_time_ms=avg_queue_time,
            thermal_throttling_events=self._thermal_throttling_events,
            memory_pressure_events=self._memory_pressure_events
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