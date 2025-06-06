#!/usr/bin/env python3
"""
Ollama Integration Example
Shows how to integrate real Ollama LLM requests with the monitoring framework.
Note: Requires Ollama to be running with a loaded model.
"""

import requests
import time
import uuid
import psutil
from datetime import datetime, timezone

class OllamaMonitoringIntegration:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.monitor_url = "http://localhost:8000"
    
    def make_monitored_ollama_request(self, model_name, prompt):
        """Make an Ollama request and track it in the monitoring framework."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        print(f"🧠 Making Ollama request to {model_name}...")
        
        try:
            # Real Ollama API call
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Extract real metrics from Ollama response
            response_text = result.get('response', '')
            response_time_ms = (end_time - start_time) * 1000
            
            # Real token counts from Ollama
            prompt_tokens = result.get('prompt_eval_count', len(prompt.split()) * 1.3)
            completion_tokens = result.get('eval_count', len(response_text.split()) * 1.3)
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Calculate tokens per second from real timing
            eval_duration_ms = result.get('eval_duration', 0) / 1_000_000
            tokens_per_second = completion_tokens / (eval_duration_ms / 1000) if eval_duration_ms > 0 else 0
            
            print(f"✅ Ollama response: {response_text[:100]}...")
            print(f"📊 Metrics: {response_time_ms:.1f}ms, {tokens_per_second:.1f} tokens/sec")
            
            # Create monitoring data with real metrics
            inference_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'request_id': request_id,
                'model_name': model_name,
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': total_tokens,
                'response_time_ms': response_time_ms,
                'tokens_per_second': tokens_per_second,
                'success': True,
                'error_message': None,
                'memory_peak_mb': max(start_memory, end_memory),
                'gpu_utilization_percent': 0.0,  # Ollama doesn't expose this
                'cache_hit': False,
                'queue_time_ms': result.get('load_duration', 0) / 1_000_000
            }
            
            # Track in monitoring framework
            track_response = requests.post(
                f"{self.monitor_url}/track/inference",
                json=inference_data
            )
            
            if track_response.status_code == 200:
                print(f"✅ Successfully tracked in monitoring framework")
            else:
                print(f"❌ Failed to track: {track_response.status_code}")
            
            return response_text, inference_data
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            print(f"❌ Ollama request failed: {e}")
            
            # Track failure
            failure_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'request_id': request_id,
                'model_name': model_name,
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': 0,
                'total_tokens': len(prompt.split()),
                'response_time_ms': response_time_ms,
                'tokens_per_second': 0.0,
                'success': False,
                'error_message': str(e),
                'memory_peak_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'gpu_utilization_percent': 0.0,
                'cache_hit': False,
                'queue_time_ms': 0.0
            }
            
            # Track failure in monitoring
            requests.post(f"{self.monitor_url}/track/inference", json=failure_data)
            return None, failure_data

def example_usage():
    """Example of how to use Ollama with monitoring."""
    print("🚀 Ollama + Monitoring Integration Example")
    print("=" * 50)
    
    integration = OllamaMonitoringIntegration()
    
    # Check if Ollama is available
    try:
        models_response = requests.get("http://localhost:11434/api/tags")
        models = models_response.json().get('models', [])
        
        if not models:
            print("❌ No Ollama models available. Install one with:")
            print("   ollama pull llama2:7b  # or another smaller model")
            return
        
        model_name = models[0]['name']
        print(f"🎯 Using model: {model_name}")
        
        # Make monitored requests
        prompts = [
            "Hello, how are you?",
            "What is Python programming?",
            "Explain machine learning briefly."
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🔄 Request {i}/{len(prompts)}: {prompt}")
            response, metrics = integration.make_monitored_ollama_request(model_name, prompt)
            time.sleep(1)  # Brief pause
        
        # Show dashboard results
        dashboard_response = requests.get("http://localhost:8000/metrics/current")
        if dashboard_response.status_code == 200:
            data = dashboard_response.json()
            print(f"\n📊 Dashboard Summary:")
            print(f"   Total Requests: {data['performance']['total_requests']}")
            print(f"   Success Rate: {100 - data['performance']['error_rate']:.1f}%")
            print(f"   Avg Response Time: {data['performance']['avg_response_time_ms']:.1f}ms")
            print(f"   🌐 View dashboard: http://localhost:8080")
        
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("Start Ollama with: ollama serve")

if __name__ == "__main__":
    example_usage() 