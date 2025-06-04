#!/usr/bin/env python3
"""
Test Multiple Ollama Models for Performance Comparison
"""

import asyncio
from test_ollama_llm import OllamaLLMTester

async def compare_models():
    """Compare performance across different Ollama models."""
    
    # Available models
    models_to_test = [
        "stable-code:latest",      # 3B coding model
        "mistral-small3.1:latest", # 24B general model (if you want to test)
    ]
    
    test_prompt = "Write a Python function to sort a list of numbers:"
    
    print("🔬 Comparing Ollama Models")
    print("=" * 50)
    
    results = {}
    
    for model in models_to_test:
        print(f"\n🧪 Testing {model}...")
        
        tester = OllamaLLMTester(model_name=model)
        
        try:
            if await tester.check_ollama_status():
                result = await tester.generate_with_ollama(
                    test_prompt, 
                    max_tokens=100, 
                    temperature=0.5
                )
                
                results[model] = {
                    "success": True,
                    "response_time": result['response_time'],
                    "tokens_per_second": result['tokens_per_second'],
                    "completion_tokens": result['completion_tokens'],
                    "response_preview": result['response'][:100]
                }
                
                print(f"   ✅ {result['completion_tokens']} tokens in {result['response_time']:.2f}s")
                print(f"   📊 {result['tokens_per_second']:.1f} tokens/sec")
                
            else:
                results[model] = {"success": False, "error": "Model not available"}
                print(f"   ❌ Model not available")
                
        except Exception as e:
            results[model] = {"success": False, "error": str(e)}
            print(f"   ❌ Error: {e}")
        
        finally:
            await tester.close()
    
    # Summary comparison
    print(f"\n📊 Model Comparison Summary:")
    print("-" * 40)
    
    for model, result in results.items():
        if result.get("success"):
            print(f"{model}:")
            print(f"  Response Time: {result['response_time']:.2f}s")
            print(f"  Tokens/sec: {result['tokens_per_second']:.1f}")
            print(f"  Tokens: {result['completion_tokens']}")
        else:
            print(f"{model}: ❌ {result.get('error', 'Failed')}")
    
    print(f"\n📊 All data visible at: http://localhost:8080")

if __name__ == "__main__":
    asyncio.run(compare_models()) 