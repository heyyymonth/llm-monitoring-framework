import time
import psutil
import threading
import logging
import subprocess
import platform
from collections import deque, defaultdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import subprocess
import platform
from .models import SystemMetrics, InferenceMetrics, LLMProcessMetrics, PerformanceSummary
import subprocess
import platform

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Minimalist LLM performance monitoring."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread = None
        self.start_time = datetime.now(timezone.utc)
        
        # Metrics storage (keep last 1000 of each)
        self._system_metrics = deque(maxlen=1000)
        self._inference_metrics = deque(maxlen=1000)
        
        # Performance counters
        self._pending_requests = 0
        self._processing_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        
        # Process tracking
        self._current_process = psutil.Process()
        
    def start(self):
        """Start metrics collection."""
        if not self._running:
            self._running = True
            self._collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self._collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                system_metrics = self._collect_system_metrics()
                if system_metrics:
                    with self._lock:
                        self._system_metrics.append(system_metrics)
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect basic system metrics relevant to LLM performance."""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # LLM process metrics
            process_metrics = self._collect_llm_process_metrics()
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                system_load_1m=0.0,  # Simplified
                disk_usage_percent=0.0,  # Simplified
                network_io_mbps=0.0,  # Simplified
                llm_process=process_metrics
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    


    def _collect_llm_process_metrics(self) -> Optional[LLMProcessMetrics]:
        """Collect LLM process-specific metrics."""
        try:
            memory_info = self._current_process.memory_info()
            
            return LLMProcessMetrics(
                pid=self._current_process.pid,
                cpu_percent=self._current_process.cpu_percent(),
                memory_rss_mb=memory_info.rss / (1024 * 1024),
                memory_vms_mb=memory_info.vms / (1024 * 1024),
                memory_percent=self._current_process.memory_percent(),
                model_memory_mb=memory_info.rss / (1024 * 1024),  # Simplified estimate
                inference_threads=self._current_process.num_threads(),
                gpu_memory_mb=0.0,  # Simplified
                gpu_utilization_percent=0.0,  # Simplified
                model_loading_time_ms=0.0,  # Not tracked
                peak_memory_mb=memory_info.rss / (1024 * 1024)
            )
        except Exception as e:
            logger.warning(f"Error collecting LLM process metrics: {e}")
            return None
    
    def log_inference(self, metrics: InferenceMetrics):
        """Log inference metrics."""
        with self._lock:
            self._inference_metrics.append(metrics)
            
            if metrics.success:
                self._completed_requests += 1
            else:
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
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        with self._lock:
            return self._system_metrics[-1] if self._system_metrics else None
    
    def get_recent_inference_metrics(self, limit: int = 100) -> List[InferenceMetrics]:
        """Get recent inference metrics."""
        with self._lock:
            return list(self._inference_metrics)[-limit:]
    
    def get_performance_summary(self, time_period: str = "1h") -> PerformanceSummary:
        """Get performance summary for a time period."""
        
        # Parse time period (simplified)
        hours = 1
        if time_period == "24h":
            hours = 24
        elif time_period == "7d":
            hours = 168
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._lock:
            # Filter metrics by time period - fix datetime comparison
            recent_inferences = []
            for m in self._inference_metrics:
                # Ensure both timestamps are timezone-aware
                m_timestamp = m.timestamp
                if m_timestamp.tzinfo is None:
                    m_timestamp = m_timestamp.replace(tzinfo=timezone.utc)
                
                if m_timestamp >= cutoff:
                    recent_inferences.append(m)
        
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
        
        # Calculate basic metrics
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
        
        avg_memory_usage = sum(m.memory_peak_mb for m in recent_inferences) / len(recent_inferences) if recent_inferences else 0.0
        peak_memory_usage = max((m.memory_peak_mb for m in recent_inferences), default=0.0)
        avg_gpu = sum(m.gpu_utilization_percent for m in recent_inferences) / len(recent_inferences) if recent_inferences else 0.0
        
        cache_hits = sum(1 for m in recent_inferences if m.cache_hit)
        cache_hit_rate = (cache_hits / len(recent_inferences)) * 100 if recent_inferences else 0.0
        
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
            avg_queue_time_ms=0.0,  # Simplified
            thermal_throttling_events=0,  # Simplified
            memory_pressure_events=0  # Simplified
        )
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic stats."""
        with self._lock:
            return {
                "uptime_seconds": self.get_uptime(),
                "total_inference_requests": len(self._inference_metrics),
                "pending_requests": self._pending_requests,
                "processing_requests": self._processing_requests,
                "completed_requests": self._completed_requests,
                "failed_requests": self._failed_requests,
                "system_metrics_count": len(self._system_metrics)
            } 