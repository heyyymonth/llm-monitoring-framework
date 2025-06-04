#!/usr/bin/env python3
"""
Example integrations showing how to use the LLM monitoring framework
with different LLM libraries and services.
"""

import asyncio
import time
import random
from typing import List
from monitoring.client import LLMMonitor


# =============================================================================
# OpenAI Integration Example
# =============================================================================

async def openai_example():
    """Example integration with OpenAI API."""
    print("🔄 OpenAI Integration Example")
    
    # Note: This requires openai package: pip install openai
    try:
        import openai
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai")
        return
    
    # Initialize monitor
    monitor = LLMMonitor("http://localhost:8000")
    
    # Initialize OpenAI client (you'll need to set your API key)
    # openai.api_key = "your-api-key-here"
    
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How does machine learning work?"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Processing request {i+1}/{len(prompts)}")
        
        with monitor.track_request(model_name="gpt-3.5-turbo") as tracker:
            try:
                # Set prompt info
                prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
                tracker.set_prompt_info(
                    tokens=int(prompt_tokens),
                    length=len(prompt),
                    temperature=0.7,
                    max_tokens=150
                )
                
                # Simulate API call (replace with actual OpenAI call)
                tracker.start_processing()
                
                # Simulate processing time
                processing_time = random.uniform(0.5, 3.0)
                await asyncio.sleep(processing_time)
                
                # Simulate response
                response_text = f"This is a simulated response to: {prompt[:50]}..."
                completion_tokens = len(response_text.split()) * 1.3
                
                tracker.set_response_info(
                    tokens=int(completion_tokens),
                    length=len(response_text)
                )
                
                # Add metadata
                tracker.set_metadata(
                    model_version="gpt-3.5-turbo-0613",
                    finish_reason="stop"
                )
                
                print(f"✅ Completed: {prompt[:30]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    await monitor.close()
    print("✅ OpenAI example completed")


# =============================================================================
# Hugging Face Transformers Example
# =============================================================================

async def huggingface_example():
    """Example integration with Hugging Face Transformers."""
    print("🔄 Hugging Face Transformers Example")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("❌ Transformers package not installed. Run: pip install transformers torch")
        return
    
    # Initialize monitor
    monitor = LLMMonitor("http://localhost:8000")
    
    # Simulate model loading (replace with actual model)
    model_name = "gpt2"  # Small model for demo
    print(f"Loading model: {model_name}")
    
    prompts = [
        "The future of artificial intelligence",
        "Climate change and renewable energy",
        "The importance of education",
        "Technology in healthcare",
        "Space exploration adventures"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Processing request {i+1}/{len(prompts)}")
        
        with monitor.track_request(model_name=model_name) as tracker:
            try:
                # Simulate tokenization
                prompt_tokens = len(prompt.split()) + random.randint(5, 15)
                
                tracker.set_prompt_info(
                    tokens=prompt_tokens,
                    length=len(prompt),
                    temperature=0.8,
                    max_tokens=100
                )
                
                # Simulate queue time
                await asyncio.sleep(random.uniform(0.01, 0.1))
                tracker.start_processing()
                
                # Simulate model inference
                processing_time = random.uniform(1.0, 4.0)
                await asyncio.sleep(processing_time)
                
                # Simulate response generation
                completion_tokens = random.randint(20, 100)
                response_length = completion_tokens * 4  # Rough estimate
                
                tracker.set_response_info(
                    tokens=completion_tokens,
                    length=response_length
                )
                
                # Add model-specific metadata
                tracker.set_metadata(
                    model_size="124M",
                    device="cpu",
                    torch_version=torch.__version__
                )
                
                print(f"✅ Generated {completion_tokens} tokens")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        await asyncio.sleep(0.3)
    
    await monitor.close()
    print("✅ Hugging Face example completed")


# =============================================================================
# FastAPI Integration Example
# =============================================================================

def create_fastapi_app():
    """Example of integrating monitoring into a FastAPI LLM service."""
    
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi")
        return None
    
    app = FastAPI(title="Monitored LLM Service")
    monitor = LLMMonitor("http://localhost:8000")
    
    class CompletionRequest(BaseModel):
        prompt: str
        max_tokens: int = 100
        temperature: float = 0.7
        model: str = "default-model"
    
    class CompletionResponse(BaseModel):
        text: str
        tokens_used: int
        response_time_ms: float
    
    @app.post("/v1/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        """Generate text completion with monitoring."""
        
        with monitor.track_request(
            model_name=request.model
        ) as tracker:
            try:
                # Set prompt info
                prompt_tokens = len(request.prompt.split()) + 10
                tracker.set_prompt_info(
                    tokens=prompt_tokens,
                    length=len(request.prompt),
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                # Simulate queue time
                await asyncio.sleep(random.uniform(0.01, 0.05))
                tracker.start_processing()
                
                # Simulate model inference
                processing_time = random.uniform(0.5, 2.0)
                await asyncio.sleep(processing_time)
                
                # Generate response
                response_text = f"Generated response for: {request.prompt[:30]}..."
                completion_tokens = random.randint(20, request.max_tokens)
                
                tracker.set_response_info(
                    tokens=completion_tokens,
                    length=len(response_text)
                )
                
                # Add request metadata
                tracker.set_metadata(
                    endpoint="/v1/completions",
                    user_agent="example-client"
                )
                
                return CompletionResponse(
                    text=response_text,
                    tokens_used=prompt_tokens + completion_tokens,
                    response_time_ms=tracker.start_time * 1000
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    return app


# =============================================================================
# Custom Model Integration Example
# =============================================================================

class CustomLLMWrapper:
    """Example wrapper for a custom LLM with monitoring."""
    
    def __init__(self, model_name: str, monitor_url: str = "http://localhost:8000"):
        self.model_name = model_name
        self.monitor = LLMMonitor(monitor_url)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with monitoring."""
        
        with self.monitor.track_request(self.model_name) as tracker:
            try:
                # Extract parameters
                max_tokens = kwargs.get('max_tokens', 100)
                temperature = kwargs.get('temperature', 0.7)
                
                # Set prompt info
                prompt_tokens = self._count_tokens(prompt)
                tracker.set_prompt_info(
                    tokens=prompt_tokens,
                    length=len(prompt),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Simulate processing
                tracker.start_processing()
                response = await self._run_inference(prompt, **kwargs)
                
                # Set response info
                completion_tokens = self._count_tokens(response)
                tracker.set_response_info(
                    tokens=completion_tokens,
                    length=len(response)
                )
                
                return response
                
            except Exception as e:
                # Error will be automatically logged by tracker
                raise
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting (replace with actual tokenizer)."""
        return len(text.split()) + random.randint(5, 15)
    
    async def _run_inference(self, prompt: str, **kwargs) -> str:
        """Simulate model inference."""
        # Simulate processing time based on prompt length
        processing_time = len(prompt) * 0.01 + random.uniform(0.5, 2.0)
        await asyncio.sleep(processing_time)
        
        # Generate simulated response
        max_tokens = kwargs.get('max_tokens', 100)
        response_length = random.randint(max_tokens // 2, max_tokens)
        
        return f"Generated response ({response_length} tokens) for prompt: {prompt[:30]}..."


async def custom_model_example():
    """Example using the custom model wrapper."""
    print("🔄 Custom Model Integration Example")
    
    model = CustomLLMWrapper("custom-llm-v1")
    
    prompts = [
        "Write a story about a robot",
        "Explain the water cycle",
        "List healthy meal ideas",
        "Describe ancient civilizations",
        "Compare different programming languages"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Processing request {i+1}/{len(prompts)}")
        
        try:
            response = await model.generate(
                prompt,
                max_tokens=80,
                temperature=0.8
            )
            print(f"✅ Generated: {response[:50]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(0.5)
    
    await model.monitor.close()
    print("✅ Custom model example completed")


# =============================================================================
# Main Example Runner
# =============================================================================

async def run_all_examples():
    """Run all integration examples."""
    print("🚀 Starting LLM Monitoring Integration Examples")
    print("=" * 60)
    
    examples = [
        ("OpenAI Integration", openai_example),
        ("Hugging Face Integration", huggingface_example),
        ("Custom Model Integration", custom_model_example),
    ]
    
    for name, example_func in examples:
        print(f"\n📋 Running: {name}")
        print("-" * 40)
        
        try:
            await example_func()
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        
        print()
    
    print("🎉 All examples completed!")
    print("\n💡 Tips:")
    print("- View metrics at: http://localhost:8080")
    print("- API docs at: http://localhost:8000/docs")
    print("- Check the dashboard for real-time monitoring")


if __name__ == "__main__":
    print("🔧 LLM Monitoring Framework - Integration Examples")
    print("\n⚠️  Make sure the monitoring server is running:")
    print("   python main.py")
    print("\nStarting examples in 3 seconds...")
    
    time.sleep(3)
    asyncio.run(run_all_examples()) 