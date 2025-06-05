import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging
import threading

import requests

from .models import InferenceMetrics, ErrorMetrics

logger = logging.getLogger(__name__)


class InferenceTracker:
    """Tracks metrics for a single inference request."""
    
    def __init__(self, client: 'LLMMonitor', model_name: str):
        self.client = client
        self.request_id = str(uuid.uuid4())
        self.model_name = model_name
        self.start_time = time.time()
        self.end_time = None
        self.processing_start_time = None
        self.queue_time_ms = 0
        self.processing_time_ms = 0
        self.response_time_ms = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_length = 0
        self.response_length = 0
        self.tokens_per_second = 0.0
        self.success = True
        self.error_message = None
        self.error = None
        self.temperature = None
        self.max_tokens = None
        self.metadata = {}
        self.logged = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log_error(str(exc_val), exc_type.__name__)
        self.log_completion()
    
    def start_processing(self):
        """Mark the start of processing (after queue time)."""
        self.processing_start_time = time.time()
        self.queue_time_ms = (self.processing_start_time - self.start_time) * 1000
    
    def set_prompt_info(self, tokens: int, length: int, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """Set prompt information."""
        self.prompt_tokens = tokens
        self.prompt_length = length
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def set_response_info(self, tokens: int, length: int):
        """Set response information."""
        self.completion_tokens = tokens
        self.response_length = length
    
    def set_metadata(self, **kwargs):
        """Set additional metadata."""
        self.metadata.update(kwargs)
    
    def log_error(self, error_message: str, error_type: str = "UnknownError"):
        """Log an error that occurred during inference."""
        self.success = False
        self.error_message = error_message
        self.error_type = error_type
        
        # Create error metrics
        error_metrics = ErrorMetrics(
            request_id=self.request_id,
            error_type=error_type,
            error_message=error_message,
            model_name=self.model_name
        )
        
        # Send error metrics in background thread
        threading.Thread(
            target=self.client._send_error_metrics_sync,
            args=(error_metrics,),
            daemon=True
        ).start()
    
    def log_completion(self):
        """Log the completion of the request."""
        if self.logged:
            return
            
        self.end_time = time.time()
        self.response_time_ms = (self.end_time - self.start_time) * 1000
        
        # Calculate processing time if we have it
        if self.processing_start_time:
            self.processing_time_ms = (self.end_time - self.processing_start_time) * 1000
        else:
            self.processing_time_ms = self.response_time_ms - self.queue_time_ms
        
        # Calculate tokens per second
        if self.response_time_ms > 0 and self.completion_tokens > 0:
            self.tokens_per_second = (self.completion_tokens / self.response_time_ms) * 1000
        
        # Create metrics
        inference_metrics = InferenceMetrics(
            request_id=self.request_id,
            model_name=self.model_name,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            response_time_ms=self.response_time_ms,
            queue_time_ms=self.queue_time_ms,
            processing_time_ms=self.processing_time_ms,
            tokens_per_second=self.tokens_per_second,
            prompt_length=self.prompt_length,
            response_length=self.response_length,
            success=self.success,
            error_message=self.error_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=self.metadata
        )
        
        # Send metrics in background thread
        thread = threading.Thread(
            target=self.client._send_inference_metrics_sync,
            args=(inference_metrics,),
            daemon=True
        )
        thread.start()
        
        # Give thread a moment to start and complete
        time.sleep(0.1)
        
        self.logged = True


class LLMMonitor:
    """Client for monitoring LLM inference performance."""
    
    def __init__(self, monitor_url: str = "http://localhost:8000", timeout: float = 5.0):
        self.monitor_url = monitor_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    def _send_inference_metrics_sync(self, inference_metrics: InferenceMetrics):
        """Send inference metrics synchronously."""
        try:
            # Convert metrics to dict with JSON-serializable datetime
            inference_dict = inference_metrics.model_dump()
            inference_dict['timestamp'] = inference_metrics.timestamp.isoformat()
            
            # Send inference metrics
            response = self.session.post(
                f"{self.monitor_url}/track/inference",
                json=inference_dict
            )
            response.raise_for_status()
            logger.debug(f"Successfully sent inference metrics for request {inference_metrics.request_id}")
            
        except Exception as e:
            logger.warning(f"Failed to send inference metrics: {e}")
    
    def _send_error_metrics_sync(self, error_metrics: ErrorMetrics):
        """Send error metrics synchronously."""
        try:
            # Convert metrics to dict with JSON-serializable datetime
            error_dict = error_metrics.model_dump()
            error_dict['timestamp'] = error_metrics.timestamp.isoformat()
            
            # Send error metrics
            response = self.session.post(
                f"{self.monitor_url}/track/error",
                json=error_dict
            )
            response.raise_for_status()
            logger.debug(f"Successfully sent error metrics for request {error_metrics.request_id}")
            
        except Exception as e:
            logger.warning(f"Failed to send error metrics: {e}")
    
    def track_inference(self, model_name: Optional[str] = None, request_id: Optional[str] = None) -> InferenceTracker:
        """Create a tracker for monitoring an inference request."""
        return InferenceTracker(self, model_name)
    
    @contextmanager
    def track_request(self, model_name: Optional[str] = None, request_id: Optional[str] = None):
        """Context manager for tracking inference requests."""
        tracker = self.track_inference(model_name, request_id)
        try:
            yield tracker
        except Exception as e:
            tracker.log_error(str(e), type(e).__name__)
            raise
        else:
            tracker.log_completion()
    
    def send_custom_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Send a custom metric to the monitoring server."""
        try:
            payload = {
                "metric_name": metric_name,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            response = self.session.post(
                f"{self.monitor_url}/track/custom",
                json=payload
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.warning(f"Failed to send custom metric: {e}")
    
    def get_health_status(self) -> Optional[Dict[str, Any]]:
        """Get health status from monitoring server."""
        try:
            response = self.session.get(f"{self.monitor_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get health status: {e}")
            return None
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current metrics from monitoring server."""
        try:
            response = self.session.get(f"{self.monitor_url}/metrics/current")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get current metrics: {e}")
            return None
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()


# Convenience functions for quick integration
def create_monitor(monitor_url: str = "http://localhost:8000") -> LLMMonitor:
    """Create a new LLM monitor instance."""
    return LLMMonitor(monitor_url)


@contextmanager
def track_llm_call(monitor: LLMMonitor, model_name: Optional[str] = None, request_id: Optional[str] = None):
    """Simple context manager for tracking LLM calls."""
    with monitor.track_request(model_name, request_id) as tracker:
        yield tracker 