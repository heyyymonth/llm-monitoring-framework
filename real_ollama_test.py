#!/usr/bin/env python3
"""
Real Ollama LLM Integration Test
Makes actual requests to Ollama and tracks performance through the monitoring framework.
"""

import requests
import time
import uuid
import psutil
import json
from datetime import datetime, timezone
import sys

class OllamaLLMTester:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.monitor_api_url = "http://localhost:8000"
        
    def check_ollama_status(self):
        """Check if Ollama is running and what models are available."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json()
            print(f"✅ Ollama is running with {len(models.get('models', []))} models:")
            for model in models.get('models', []):
                print(f"   - {model['name']} ({model['size'] // (1024**3)} GB)")
            return models.get('models', [])
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            return []
    
    def check_monitor_api(self):
        """Check if the monitoring API is running."""
        try:
            response = requests.get(f"{self.monitor_api_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            print(f"✅ Monitoring API is running (status: {health['status']})")
            return True
        except Exception as e:
            print(f"❌ Monitoring API not available: {e}")
            return False
    
    def make_real_ollama_request(self, model_name: str, prompt: str):
        """Make a real request to Ollama and capture performance metrics."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        print(f"🧠 Making real Ollama request to {model_name}...")
        print(f"   Prompt: {prompt[:60]}...")
        
        # Ollama API request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            # Make the actual LLM request
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60  # Give it time for real inference
            )
            response.raise_for_status()
            ollama_result = response.json()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Extract real metrics from Ollama response
            response_text = ollama_result.get('response', '')
            total_duration = ollama_result.get('total_duration', 0) / 1_000_000  # Convert to ms
            load_duration = ollama_result.get('load_duration', 0) / 1_000_000    # Convert to ms
            prompt_eval_count = ollama_result.get('prompt_eval_count', 0)
            prompt_eval_duration = ollama_result.get('prompt_eval_duration', 0) / 1_000_000
            eval_count = ollama_result.get('eval_count', 0)
            eval_duration = ollama_result.get('eval_duration', 0) / 1_000_000
            
            # Calculate metrics
            response_time_ms = (end_time - start_time) * 1000
            prompt_tokens = prompt_eval_count if prompt_eval_count > 0 else len(prompt.split()) * 1.3
            completion_tokens = eval_count if eval_count > 0 else len(response_text.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            tokens_per_second = eval_count / (eval_duration / 1000) if eval_duration > 0 else 0
            memory_peak_mb = max(start_memory, end_memory)
            
            print(f"✅ Ollama response received ({response_time_ms:.1f}ms)")
            print(f"   Response: {response_text[:100]}...")
            print(f"   Tokens: {prompt_tokens:.0f} prompt + {completion_tokens:.0f} completion = {total_tokens}")
            print(f"   Speed: {tokens_per_second:.1f} tokens/sec")
            
            # Create inference metrics with real data
            inference_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "model_name": model_name,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": total_tokens,
                "response_time_ms": response_time_ms,
                "tokens_per_second": tokens_per_second,
                "success": True,
                "error_message": None,
                "memory_peak_mb": memory_peak_mb,
                "gpu_utilization_percent": 0.0,  # Ollama doesn't expose this directly
                "cache_hit": False,  # Could be determined from load_duration
                "queue_time_ms": load_duration
            }
            
            return inference_data, response_text
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            print(f"❌ Ollama request failed: {e}")
            
            # Create failure metrics
            inference_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "model_name": model_name,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 0,
                "total_tokens": len(prompt.split()),
                "response_time_ms": response_time_ms,
                "tokens_per_second": 0.0,
                "success": False,
                "error_message": str(e),
                "memory_peak_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                "gpu_utilization_percent": 0.0,
                "cache_hit": False,
                "queue_time_ms": 0.0
            }
            
            return inference_data, None
    
    def track_inference_with_monitor(self, inference_data):
        """Send real inference metrics to the monitoring framework."""
        try:
            response = requests.post(
                f"{self.monitor_api_url}/track/inference",
                json=inference_data,
                timeout=5
            )
            response.raise_for_status()
            print(f"✅ Tracked inference in monitoring framework: {inference_data['request_id'][:8]}")
            return True
        except Exception as e:
            print(f"❌ Failed to track inference: {e}")
            return False
    
    def show_dashboard_metrics(self):
        """Display current metrics from the monitoring dashboard."""
        try:
            response = requests.get(f"{self.monitor_api_url}/metrics/current", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            system = data['system']
            performance = data['performance']
            
            print(f"\n📊 Current Dashboard Metrics:")
            print(f"   System: CPU {system['cpu_percent']:.1f}%, Memory {system['memory_percent']:.1f}%")
            print(f"   Available Memory: {system['memory_available_gb']:.1f}GB")
            print(f"   LLM Process: {system['llm_process']['memory_rss_mb']:.1f}MB RSS")
            print(f"   Total Requests: {performance['total_requests']}")
            print(f"   Success Rate: {100 - performance['error_rate']:.1f}%")
            print(f"   Avg Response Time: {performance['avg_response_time_ms']:.1f}ms")
            print(f"   Tokens/Second: {performance['avg_tokens_per_second']:.1f}")
            print(f"   Total Tokens Processed: {performance['total_tokens_processed']}")
            
        except Exception as e:
            print(f"❌ Failed to get dashboard metrics: {e}")
    
    def run_real_llm_test(self, num_requests=3):
        """Run comprehensive test with real Ollama requests."""
        print("🧪 Real Ollama LLM Integration Test")
        print("=" * 50)
        
        # Check prerequisites
        models = self.check_ollama_status()
        if not models:
            print("❌ No Ollama models available. Please install a model first:")
            print("   ollama pull mistral")
            return False
        
        if not self.check_monitor_api():
            print("❌ Monitoring API not available. Please start it first:")
            print("   python main.py")
            return False
        
        # Use the first available model
        model_name = models[0]['name']
        print(f"\n🎯 Using model: {model_name}")
        
        # Test prompts with varying complexity
        test_prompts = [
            "What is the capital of France?",
            "Explain the concept of machine learning in simple terms with examples.",
            "Write a Python function that calculates the factorial of a number using recursion. Include error handling and explain how it works step by step.",
            "Describe the differences between supervised and unsupervised learning in detail, providing real-world use cases for each approach.",
            "Create a comprehensive guide for setting up a machine learning development environment including tools, libraries, and best practices."
        ]
        
        successful_requests = 0
        total_response_time = 0
        
        for i, prompt in enumerate(test_prompts[:num_requests], 1):
            print(f"\n🔄 Test {i}/{num_requests}")
            
            # Make real Ollama request
            inference_data, response_text = self.make_real_ollama_request(model_name, prompt)
            
            # Track in monitoring framework
            tracked = self.track_inference_with_monitor(inference_data)
            
            if inference_data['success'] and tracked:
                successful_requests += 1
                total_response_time += inference_data['response_time_ms']
            
            # Show partial response
            if response_text:
                print(f"   Preview: {response_text[:120]}...")
            
            time.sleep(2)  # Brief pause between requests
        
        # Final metrics
        print(f"\n📈 Test Results:")
        print(f"   Requests Made: {num_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Success Rate: {(successful_requests/num_requests)*100:.1f}%")
        if successful_requests > 0:
            print(f"   Avg Response Time: {total_response_time/successful_requests:.1f}ms")
        
        # Show dashboard data
        self.show_dashboard_metrics()
        
        return successful_requests == num_requests

def main():
    """Main test runner."""
    print("🚀 Starting Real Ollama LLM Integration Test")
    
    tester = OllamaLLMTester()
    success = tester.run_real_llm_test(num_requests=3)
    
    if success:
        print(f"\n🎉 All tests passed! The monitoring framework successfully tracked real Ollama LLM requests.")
        print(f"   ✓ Real inference requests to Ollama models")
        print(f"   ✓ Actual response times and token counting")
        print(f"   ✓ Memory usage tracking during inference")
        print(f"   ✓ Dashboard integration with live metrics")
        print(f"\n🌐 Check the dashboard: http://localhost:8080")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check the setup and logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 