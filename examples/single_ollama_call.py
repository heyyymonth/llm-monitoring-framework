#!/usr/bin/env python3
"""
Single Ollama Call with LLM Monitoring

Quick example showing how to make one monitored Ollama call.
"""

import subprocess
import sys
import os
import time
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.client import create_monitor

def extract_tokens_estimate(text):
    """Rough estimate of tokens from text (1 token ~= 4 characters)."""
    return max(1, len(text) // 4)

def quick_ollama_call(prompt, model="stable-code"):
    """Make a single Ollama call with monitoring."""
    
    print(f"🚀 Calling Ollama with monitoring...")
    print(f"📝 Prompt: {prompt}")
    print(f"🤖 Model: {model}")
    
    monitor = create_monitor("http://localhost:8000")
    
    with monitor.track_request(model_name=model) as tracker:
        # Set prompt info
        prompt_tokens = extract_tokens_estimate(prompt)
        tracker.set_prompt_info(tokens=prompt_tokens, length=len(prompt))
        print(f"📊 Prompt tokens estimated: {prompt_tokens}")
        
        # Start processing
        tracker.start_processing()
        start_time = time.time()
        
        try:
            # Call Ollama
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Set response info
                completion_tokens = extract_tokens_estimate(response)
                tracker.set_response_info(tokens=completion_tokens, length=len(response))
                
                # Add metadata
                tracker.set_metadata(
                    duration_seconds=duration,
                    command_successful=True,
                    response_preview=response[:100] + "..." if len(response) > 100 else response
                )
                
                print(f"✅ Success! Duration: {duration:.2f}s")
                print(f"📊 Response tokens estimated: {completion_tokens}")
                print(f"🔤 Response preview: {response[:200]}...")
                print(f"📈 Metrics should be sent to monitoring system!")
                
                return response
            else:
                error_msg = result.stderr or "Unknown error"
                tracker.log_error(error_msg, "OllamaError")
                print(f"❌ Ollama failed: {error_msg}")
                return None
                
        except subprocess.TimeoutExpired:
            tracker.log_error("Request timeout", "TimeoutError")
            print("❌ Request timed out")
            return None
        except Exception as e:
            tracker.log_error(str(e), type(e).__name__)
            print(f"❌ Error: {e}")
            return None

def main():
    """Main function."""
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a simple function to reverse a string"
    
    print("="*60)
    print("🔧 Testing Ollama Integration with LLM Monitoring")
    print("="*60)
    
    # Check if monitoring system is accessible
    monitor = create_monitor("http://localhost:8000")
    try:
        health = monitor.get_health_status()
        if health:
            print("✅ Monitoring system is accessible")
        else:
            print("⚠️  Warning: Cannot reach monitoring system")
    except Exception as e:
        print(f"⚠️  Warning: Monitoring system check failed: {e}")
    
    # Make the call
    response = quick_ollama_call(prompt)
    
    if response:
        print("\n" + "="*60)
        print("🎉 Integration test completed successfully!")
        print("🔍 Check dashboard at http://localhost:8080 to see metrics")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Integration test failed")
        print("="*60)

if __name__ == "__main__":
    main() 