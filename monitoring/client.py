import time
import uuid
import asyncio
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager
import logging

from .models import InferenceMetrics, ErrorMetrics
from .config import get_config

logger = logging.getLogger(__name__)


class InferenceTracker:
    """Context manager for tracking a single inference request."""
    
    def __init__(self, client: 'LLMMonitor', model_name: str):
        self.client = client
        self.model_name = model_name
        self.request_id = f"req_{int(time.time() * 1000)}_{id(self) % 10000}"
        self.start_time = time.time()
        self.queue_start_time = self.start_time
        self.processing_start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Request info
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_length = 0
        self.response_length = 0
        self.temperature: Optional[float] = None
        self.max_tokens: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        
        # Performance metrics
        self.response_time_ms = 0.0
        self.tokens_per_second = 0.0
        
        # Status
        self.success = True
        self.error_message: Optional[str] = None
        self.error: Optional[Exception] = None
        self.logged = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log_error(str(exc_val), exc_type.__name__)
        else:
            self.log_completion()
    
    def start_processing(self):
        """Mark the start of actual processing (end of queue time)."""
        self.processing_start_time = time.time()
    
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
        """Log an error during inference."""
        self.success = False
        self.error_message = error_message
        
        end_time = time.time()
        response_time_ms = (end_time - self.start_time) * 1000
        queue_time_ms = ((self.processing_start_time or end_time) - self.queue_start_time) * 1000
        processing_time_ms = ((end_time - self.processing_start_time) if self.processing_start_time else 0) * 1000
        
        # Log inference metrics
        inference_metrics = InferenceMetrics(
            request_id=self.request_id,
            model_name=self.model_name,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=0,  # No completion on error
            total_tokens=self.prompt_tokens,
            response_time_ms=response_time_ms,
            queue_time_ms=queue_time_ms,
            processing_time_ms=processing_time_ms,
            tokens_per_second=0.0,
            prompt_length=self.prompt_length,
            response_length=0,
            success=False,
            error_message=error_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=self.metadata
        )
        
        # Log error metrics
        error_metrics = ErrorMetrics(
            request_id=self.request_id,
            error_type=error_type,
            error_message=error_message,
            model_name=self.model_name
        )
        
        asyncio.create_task(self.client._send_metrics(inference_metrics, error_metrics))
    
    def log_completion(self):
        """Log the completion of the request."""
        if self.logged:
            return
            
        self.end_time = time.time()
        self.response_time_ms = (self.end_time - self.start_time) * 1000
        
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
            tokens_per_second=self.tokens_per_second,
            prompt_length=self.prompt_length,
            response_length=self.response_length,
            success=self.success,
            error_message=self.error_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=self.metadata
        )
        
        error_metrics = None
        if not self.success and self.error_message:
            error_metrics = ErrorMetrics(
                request_id=self.request_id,
                error_type=type(self.error).__name__ if self.error else "UnknownError",
                error_message=self.error_message,
                model_name=self.model_name
            )
        
        # Try to send metrics in background (safely handle event loop)
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If we have a loop, create task
                asyncio.create_task(self.client._send_metrics(inference_metrics, error_metrics))
            except RuntimeError:
                # No event loop running, use thread
                import threading
                thread = threading.Thread(
                    target=self._send_metrics_sync,
                    args=(inference_metrics, error_metrics)
                )
                thread.daemon = True
                thread.start()
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
        
        self.logged = True
    
    def _send_metrics_sync(self, inference_metrics, error_metrics=None):
        """Send metrics synchronously in a separate thread."""
        try:
            import asyncio
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function
            loop.run_until_complete(
                self.client._send_metrics(inference_metrics, error_metrics)
            )
            
        except Exception as e:
            logger.warning(f"Failed to send metrics in thread: {e}")
        finally:
            try:
                loop.close()
            except:
                pass


class LLMMonitor:
    """Client for monitoring LLM inference performance."""
    
    def __init__(self, monitor_url: str = "http://localhost:8000", timeout: float = 5.0):
        self.monitor_url = monitor_url.rstrip('/')
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def _send_metrics(self, inference_metrics: InferenceMetrics, error_metrics: Optional[ErrorMetrics] = None):
        """Send metrics to monitoring server."""
        try:
            client = await self._get_client()
            
            # Convert metrics to dict with JSON-serializable datetime
            inference_dict = inference_metrics.model_dump()
            inference_dict['timestamp'] = inference_metrics.timestamp.isoformat()
            
            # Send inference metrics
            await client.post(
                f"{self.monitor_url}/track/inference",
                json=inference_dict
            )
            
            # Send error metrics if present
            if error_metrics:
                error_dict = error_metrics.model_dump()
                error_dict['timestamp'] = error_metrics.timestamp.isoformat()
                await client.post(
                    f"{self.monitor_url}/track/error",
                    json=error_dict
                )
                
        except Exception as e:
            logger.warning(f"Failed to send metrics to monitoring server: {e}")
    
    def track_inference(self, model_name: Optional[str] = None, request_id: Optional[str] = None) -> InferenceTracker:
        """Create a tracker for monitoring an inference request."""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        return InferenceTracker(self, model_name)
    
    @contextmanager
    def track_request(self, model_name: Optional[str] = None, request_id: Optional[str] = None):
        """Context manager for tracking inference requests (synchronous version)."""
        tracker = self.track_inference(model_name, request_id)
        try:
            yield tracker
        except Exception as e:
            tracker.log_error(str(e), type(e).__name__)
            raise
        else:
            tracker.log_completion()
    
    async def send_custom_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Send a custom metric to the monitoring server."""
        try:
            client = await self._get_client()
            
            payload = {
                "metric_name": metric_name,
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            await client.post(
                f"{self.monitor_url}/track/custom",
                json=payload
            )
            
        except Exception as e:
            logger.warning(f"Failed to send custom metric: {e}")
    
    async def get_health_status(self) -> Optional[Dict[str, Any]]:
        """Get health status from monitoring server."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.monitor_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get health status: {e}")
            return None
    
    async def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current metrics from monitoring server."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.monitor_url}/metrics/current")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get current metrics: {e}")
            return None
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Convenience functions for quick integration
def create_monitor(monitor_url: str = "http://localhost:8000") -> LLMMonitor:
    """Create a new LLM monitor instance."""
    return LLMMonitor(monitor_url)


@contextmanager
def track_llm_call(monitor: LLMMonitor, model_name: Optional[str] = None, request_id: Optional[str] = None):
    """Simple context manager for tracking LLM calls."""
    with monitor.track_request(model_name, request_id) as tracker:
        yield tracker


# Example usage functions
async def example_openai_integration(monitor: LLMMonitor, openai_client, prompt: str):
    """Example of integrating with OpenAI API."""
    with monitor.track_request(model_name="gpt-3.5-turbo") as tracker:
        # Set prompt info
        prompt_tokens = len(prompt.split())  # Rough estimate
        tracker.set_prompt_info(
            tokens=prompt_tokens, 
            length=len(prompt),
            temperature=0.7,
            max_tokens=150
        )
        
        # Mark start of processing
        tracker.start_processing()
        
        # Make API call
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        # Set response info
        completion_tokens = response.usage.completion_tokens
        response_text = response.choices[0].message.content
        
        tracker.set_response_info(
            tokens=completion_tokens,
            length=len(response_text)
        )
        
        # Add metadata
        tracker.set_metadata(
            model_version=response.model,
            finish_reason=response.choices[0].finish_reason
        )
        
        return response_text


async def example_huggingface_integration(monitor: LLMMonitor, model, tokenizer, prompt: str):
    """Example of integrating with Hugging Face models."""
    with monitor.track_request(model_name=model.name_or_path) as tracker:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]
        
        tracker.set_prompt_info(
            tokens=prompt_tokens,
            length=len(prompt)
        )
        
        # Mark start of processing
        tracker.start_processing()
        
        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_length=prompt_tokens + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        response_tokens = outputs[0][prompt_tokens:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        tracker.set_response_info(
            tokens=len(response_tokens),
            length=len(response_text)
        )
        
        return response_text 