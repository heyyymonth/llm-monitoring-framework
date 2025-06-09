#!/usr/bin/env python3
"""
LLM Monitoring Integration Test

Tests the monitoring framework with real Ollama models to verify:
- Quality assessment
- Safety evaluation  
- Cost tracking
- API functionality
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MONITOR_BASE_URL = "http://localhost:8000"

def check_ollama_status() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return [model["name"] for model in response.json().get("models", [])]
    except:
        pass
    return []

def check_monitor_api() -> bool:
    """Check if monitoring API is running."""
    try:
        response = requests.get(f"{MONITOR_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_ollama(model: str, prompt: str) -> Dict[str, Any]:
    """Query Ollama model and return response with timing."""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            end_time = time.time()
            
            return {
                "success": True,
                "response": data.get("response", ""),
                "response_time_ms": int((end_time - start_time) * 1000),
                "tokens": {
                    "prompt": data.get("prompt_eval_count", 0),
                    "output": data.get("eval_count", 0)
                }
            }
    except Exception as e:
        print(f"Error querying Ollama: {e}")
    
    return {"success": False}

def monitor_inference(prompt: str, response: str, model: str) -> Dict[str, Any]:
    """Send inference data to monitoring API."""
    try:
        api_response = requests.post(
            f"{MONITOR_BASE_URL}/monitor/inference",
            json={
                "prompt": prompt,
                "response": response,
                "model_name": model
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if api_response.status_code == 200:
            return {"success": True, "data": api_response.json()}
        else:
            return {"success": False, "error": f"API returned {api_response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_metrics_summary() -> Dict[str, Any]:
    """Fetch current metrics from monitoring API."""
    try:
        endpoints = ["quality", "safety", "cost"]
        metrics = {}
        
        for endpoint in endpoints:
            response = requests.get(f"{MONITOR_BASE_URL}/metrics/{endpoint}", timeout=5)
            if response.status_code == 200:
                metrics[endpoint] = response.json()
        
        return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {}

def main():
    """Run the integration test."""
    print("🧪 Testing Ollama Integration with LLM Monitoring Framework")
    print("=" * 70)
    
    # Check prerequisites
    if not check_ollama_status():
        print("❌ Ollama server not available at http://localhost:11434")
        return
    
    if not check_monitor_api():
        print("❌ Monitoring API not available at http://localhost:8000")
        return
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ No Ollama models available")
        return
    
    print(f"✅ Ollama is running with {len(models)} models available")
    for model in models:
        print(f"   - {model}")
    
    print("✅ Monitoring API is running")
    
    # Use the first available model
    model_name = models[0]
    print(f"\n🚀 Using model: {model_name}")
    
    # Test prompts
    test_prompts = [
        "What is 2+2?",
        "Hello world in Python"
    ]
    
    print(f"\n📝 Testing with {len(test_prompts)} prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        
        # Query Ollama
        result = query_ollama(model_name, prompt)
        
        if not result["success"]:
            print("❌ Failed to get response from Ollama")
            continue
        
        response_text = result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"]
        print(f"Response: {response_text}")
        print(f"Response Time: {result['response_time_ms']}ms")
        print(f"Tokens - Prompt: {result['tokens']['prompt']}, Output: {result['tokens']['output']}")
        
        # Monitor the inference
        monitor_result = monitor_inference(prompt, result["response"], model_name)
        
        if monitor_result["success"]:
            data = monitor_result["data"]
            print("✅ Monitoring recorded:")
            print(f"   Trace ID: {data['trace_id']}")
            print(f"   Quality Score: {data['quality_score']:.3f}")
            print(f"   Safety Score: {data['safety_score']:.3f}")
            print(f"   Is Safe: {data['is_safe']}")
            print(f"   Safety Flags: {data['safety_flags']}")
            print(f"   Cost: ${data['cost_usd']:.6f}")
            print("✅ Successfully tracked in monitoring system")
        else:
            print(f"❌ Monitoring failed: {monitor_result.get('error', 'Unknown error')}")
    
    # Final metrics summary
    print("\n" + "=" * 70)
    print("📈 Final Metrics Summary:")
    
    print("\n📊 Fetching current metrics...")
    metrics = get_metrics_summary()
    
    if metrics:
        if "quality" in metrics:
            q = metrics["quality"]
            print("Quality Metrics:")
            print(f"   Average Quality: {q.get('average_quality', 'N/A')}")
            print(f"   Top Issues: {q.get('top_issues', [])}")
        
        if "safety" in metrics:
            s = metrics["safety"]
            print("Safety Metrics:")
            print(f"   Total Interactions: {s.get('total_interactions', 'N/A')}")
            print(f"   Safety Violations: {s.get('safety_violations', 'N/A')}")
            print(f"   Violation Rate: {s.get('violation_rate', 'N/A')}")
        
        if "cost" in metrics:
            c = metrics["cost"]
            print("Cost Metrics:")
            print(f"   Total Cost: ${c.get('total_cost_usd', 0):.6f}")
            print(f"   Avg Cost per Request: ${c.get('avg_cost_per_request', 0):.6f}")
    
    print(f"\n💡 You can view the dashboard at: http://localhost:8080")
    print(f"💡 API docs available at: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 