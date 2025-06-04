#!/usr/bin/env python3
"""
Test Ollama LLM with Performance Monitoring
Real LLM testing using Ollama API integration.
"""

import asyncio
import time
import json
import httpx
from monitoring.client import LLMMonitor

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "stable-code:latest"  # Use the available model
MONITOR_URL = "http://localhost:8000"

class OllamaLLMTester:
    """Test class for Ollama LLM with monitoring integration."""
    
    def __init__(self, model_name=MODEL_NAME, ollama_url=OLLAMA_URL):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.monitor = LLMMonitor(MONITOR_URL)
        self.client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for LLM
    
    async def check_ollama_status(self):
        """Check if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]
                print(f"✅ Ollama running. Available models: {available_models}")
                
                if self.model_name in available_models:
                    print(f"✅ Model '{self.model_name}' is available")
                    return True
                else:
                    print(f"❌ Model '{self.model_name}' not found")
                    print(f"💡 Available models: {available_models}")
                    return False
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Make sure Ollama is running with: ollama serve")
            return False
    
    async def generate_with_ollama(self, prompt, max_tokens=150, temperature=0.7):
        """Generate text using Ollama API with monitoring."""
        
        with self.monitor.track_request(model_name=self.model_name) as tracker:
            try:
                # Estimate prompt tokens (rough)
                prompt_tokens = len(prompt.split()) + len(prompt) // 4
                tracker.set_prompt_info(
                    tokens=prompt_tokens,
                    length=len(prompt),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                print(f"   🔄 Processing with {self.model_name}...")
                tracker.start_processing()
                
                # Prepare Ollama request
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }
                
                # Make request to Ollama
                start_time = time.time()
                response = await self.client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "")
                    
                    # Extract token information from Ollama response
                    eval_count = result.get("eval_count", 0)  # Output tokens
                    prompt_eval_count = result.get("prompt_eval_count", prompt_tokens)  # Input tokens
                    
                    # Calculate performance metrics
                    response_time = time.time() - start_time
                    tokens_per_second = eval_count / response_time if response_time > 0 else 0
                    
                    # Log response info
                    tracker.set_response_info(
                        tokens=eval_count,
                        length=len(generated_text)
                    )
                    
                    # Add Ollama-specific metadata
                    tracker.set_metadata(
                        model_version=self.model_name,
                        prompt_eval_count=prompt_eval_count,
                        eval_count=eval_count,
                        eval_duration_ms=result.get("eval_duration", 0) / 1000000,  # Convert to ms
                        total_duration_ms=result.get("total_duration", 0) / 1000000,
                        tokens_per_second=tokens_per_second,
                        ollama_response=True
                    )
                    
                    return {
                        "response": generated_text,
                        "prompt_tokens": prompt_eval_count,
                        "completion_tokens": eval_count,
                        "total_tokens": prompt_eval_count + eval_count,
                        "response_time": response_time,
                        "tokens_per_second": tokens_per_second
                    }
                
                else:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    tracker.log_error(error_msg, "OllamaAPIError")
                    raise Exception(error_msg)
                    
            except Exception as e:
                error_msg = f"Error generating with Ollama: {str(e)}"
                tracker.log_error(error_msg, type(e).__name__)
                raise
    
    async def run_coding_tests(self):
        """Run coding-focused tests since we're using stable-code model."""
        print("\n🧠 Testing Ollama LLM (stable-code) with Monitoring")
        print("=" * 60)
        
        coding_prompts = [
            {
                "prompt": "Write a Python function to calculate factorial:",
                "category": "basic_function"
            },
            {
                "prompt": "Create a simple REST API endpoint using FastAPI:",
                "category": "web_api"
            },
            {
                "prompt": "Write a function to reverse a string without using built-in reverse:",
                "category": "algorithm"
            },
            {
                "prompt": "Explain what this code does: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "category": "code_explanation"
            },
            {
                "prompt": "Write a Python class for a simple calculator:",
                "category": "oop"
            }
        ]
        
        results = []
        total_start_time = time.time()
        
        for i, test_case in enumerate(coding_prompts, 1):
            prompt = test_case["prompt"]
            category = test_case["category"]
            
            print(f"\n📝 Test {i}/{len(coding_prompts)}: {category}")
            print(f"   Prompt: {prompt}")
            
            try:
                result = await self.generate_with_ollama(
                    prompt, 
                    max_tokens=200, 
                    temperature=0.3  # Lower temperature for code generation
                )
                
                print(f"   ✅ Generated {result['completion_tokens']} tokens in {result['response_time']:.2f}s")
                print(f"   📊 {result['tokens_per_second']:.1f} tokens/sec")
                print(f"   💬 Response: {result['response'][:100]}...")
                
                results.append({
                    "test_case": category,
                    "success": True,
                    "tokens": result['completion_tokens'],
                    "response_time": result['response_time'],
                    "tokens_per_second": result['tokens_per_second']
                })
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results.append({
                    "test_case": category,
                    "success": False,
                    "error": str(e)
                })
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        # Summary
        total_time = time.time() - total_start_time
        successful_tests = [r for r in results if r.get("success", False)]
        
        print(f"\n" + "=" * 60)
        print(f"🎉 Testing Complete!")
        print(f"📊 Results Summary:")
        print(f"   Total Tests: {len(results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(results) - len(successful_tests)}")
        print(f"   Total Time: {total_time:.2f}s")
        
        if successful_tests:
            avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
            avg_tokens_per_sec = sum(r['tokens_per_second'] for r in successful_tests) / len(successful_tests)
            total_tokens = sum(r['tokens'] for r in successful_tests)
            
            print(f"   Average Response Time: {avg_response_time:.2f}s")
            print(f"   Average Tokens/sec: {avg_tokens_per_sec:.1f}")
            print(f"   Total Tokens Generated: {total_tokens}")
        
        print(f"\n📊 View detailed metrics at: http://localhost:8080")
        print(f"🔍 API details at: http://localhost:8000/docs")
        
        return results
    
    async def run_simple_test(self):
        """Run a simple test with one prompt."""
        print("\n🧠 Simple Ollama LLM Test")
        print("=" * 40)
        
        prompt = "Write a simple Python hello world program"
        print(f"📝 Testing with: {prompt}")
        
        try:
            result = await self.generate_with_ollama(prompt, max_tokens=100)
            
            print(f"✅ Success!")
            print(f"📊 Tokens: {result['completion_tokens']}")
            print(f"⏱️  Time: {result['response_time']:.2f}s")
            print(f"🚀 Speed: {result['tokens_per_second']:.1f} tokens/sec")
            print(f"💬 Response:\n{result['response']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()
        await self.monitor.close()

async def main():
    """Main function to run Ollama LLM tests."""
    tester = OllamaLLMTester()
    
    try:
        # Check if Ollama is ready
        print("🔍 Checking Ollama status...")
        if not await tester.check_ollama_status():
            print("\n💡 To fix this:")
            print("1. Start Ollama: ollama serve")
            print("2. Pull a model: ollama pull stable-code")
            print("3. Run this script again")
            return
        
        # Choose test type
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--simple":
            await tester.run_simple_test()
        else:
            await tester.run_coding_tests()
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        await tester.close()

if __name__ == "__main__":
    print("🦙 Ollama LLM Performance Testing")
    print("=" * 50)
    asyncio.run(main()) 