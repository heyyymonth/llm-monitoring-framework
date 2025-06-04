#!/usr/bin/env python3
"""
Test Your LLM with Performance Monitoring
Shows how to integrate any LLM with the monitoring framework.
"""

import asyncio
import time
import random
from monitoring.client import LLMMonitor

# Initialize the monitor (make sure monitoring server is running)
monitor = LLMMonitor("http://localhost:8000")

def test_example_llm():
    """Example 1: Basic LLM testing with monitoring"""
    print("🧠 Testing LLM with Performance Monitoring")
    print("=" * 50)
    
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a Python function to reverse a string",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt[:30]}...")
        
        # Use the monitoring context manager
        with monitor.track_request(model_name="my-test-llm") as tracker:
            
            # Set prompt information
            prompt_tokens = len(prompt.split()) + 5  # Rough estimate
            tracker.set_prompt_info(
                tokens=prompt_tokens,
                length=len(prompt),
                temperature=0.7,
                max_tokens=150
            )
            
            # Simulate your LLM processing
            print("   🔄 Processing...")
            tracker.start_processing()
            
            # Simulate your LLM inference time
            processing_time = random.uniform(0.5, 3.0)
            time.sleep(processing_time)
            
            # Simulate response (replace with your actual LLM call)
            response = f"This is a test response to: {prompt[:20]}... (simulated)"
            completion_tokens = len(response.split()) + 3
            
            # Log the response
            tracker.set_response_info(
                tokens=completion_tokens,
                length=len(response)
            )
            
            # Add any metadata
            tracker.set_metadata(
                model_version="1.0.0",
                temperature_used=0.7,
                test_run=True
            )
            
            print(f"   ✅ Complete! Response: {response[:50]}...")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print(f"\n🎉 Completed {len(test_prompts)} LLM tests!")
    print("📊 Check the dashboard at: http://localhost:8080")

# Example 2: Testing with Hugging Face model
def test_huggingface_llm():
    """Example for Hugging Face Transformers"""
    print("\n🤗 Hugging Face Model Example")
    print("-" * 30)
    
    # Uncomment and modify if you have transformers installed:
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "gpt2"  # or your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt = "The future of AI is"
    
    with monitor.track_request(model_name=model_name) as tracker:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]
        
        tracker.set_prompt_info(
            tokens=prompt_tokens,
            length=len(prompt)
        )
        
        tracker.start_processing()
        
        # Generate
        outputs = model.generate(
            inputs.input_ids,
            max_length=prompt_tokens + 50,
            temperature=0.8,
            do_sample=True
        )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion_tokens = len(outputs[0]) - prompt_tokens
        
        tracker.set_response_info(
            tokens=completion_tokens,
            length=len(response)
        )
    """
    
    # Simulated version for demo
    with monitor.track_request(model_name="gpt2-simulated") as tracker:
        tracker.set_prompt_info(tokens=10, length=50)
        tracker.start_processing()
        time.sleep(1.2)  # Simulate processing
        tracker.set_response_info(tokens=25, length=100)
        print("   ✅ Hugging Face model test completed!")

# Example 3: Testing OpenAI API
async def test_openai_llm():
    """Example for OpenAI API"""
    print("\n🔮 OpenAI API Example")
    print("-" * 25)
    
    # Uncomment and add your API key if using OpenAI:
    """
    import openai
    openai.api_key = "your-api-key-here"
    
    prompt = "Explain neural networks"
    
    with monitor.track_request(model_name="gpt-3.5-turbo") as tracker:
        tracker.set_prompt_info(
            tokens=len(prompt.split()) * 1.3,
            length=len(prompt),
            temperature=0.7,
            max_tokens=100
        )
        
        tracker.start_processing()
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        tracker.set_response_info(
            tokens=response.usage.completion_tokens,
            length=len(response.choices[0].message.content)
        )
    """
    
    # Simulated version for demo
    with monitor.track_request(model_name="gpt-3.5-turbo-simulated") as tracker:
        tracker.set_prompt_info(tokens=15, length=60, temperature=0.7)
        tracker.start_processing()
        await asyncio.sleep(2.0)  # Simulate API call
        tracker.set_response_info(tokens=45, length=180)
        print("   ✅ OpenAI API test completed!")

# Example 4: Custom LLM wrapper
class MyLLMWrapper:
    """Wrapper for your custom LLM with built-in monitoring"""
    
    def __init__(self, model_name="my-custom-llm"):
        self.model_name = model_name
        self.monitor = LLMMonitor("http://localhost:8000")
    
    def generate(self, prompt, **kwargs):
        """Generate text with automatic monitoring"""
        
        with self.monitor.track_request(self.model_name) as tracker:
            # Set prompt info
            prompt_tokens = self._count_tokens(prompt)
            tracker.set_prompt_info(
                tokens=prompt_tokens,
                length=len(prompt),
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 100)
            )
            
            # Your LLM processing here
            tracker.start_processing()
            
            # Replace this with your actual LLM call
            response = self._call_my_llm(prompt, **kwargs)
            
            # Log response
            completion_tokens = self._count_tokens(response)
            tracker.set_response_info(
                tokens=completion_tokens,
                length=len(response)
            )
            
            # Add any custom metadata
            tracker.set_metadata(
                model_type="custom",
                **kwargs
            )
            
            return response
    
    def _count_tokens(self, text):
        """Simple token counting - replace with your tokenizer"""
        return int(len(text.split()) * 1.3)  # Return integer
    
    def _call_my_llm(self, prompt, **kwargs):
        """Replace this with your actual LLM inference"""
        # Simulate processing time
        time.sleep(random.uniform(0.8, 2.5))
        
        # Simulate response
        return f"Generated response for: {prompt[:30]}..."

def test_custom_llm():
    """Example for custom LLM wrapper"""
    print("\n🔧 Custom LLM Wrapper Example")
    print("-" * 35)
    
    # Initialize your wrapped LLM
    my_llm = MyLLMWrapper("my-awesome-llm")
    
    # Test it
    prompts = [
        "Tell me a joke",
        "Explain blockchain",
        "Write a poem about Python"
    ]
    
    for prompt in prompts:
        print(f"   🔄 Testing: {prompt}")
        response = my_llm.generate(
            prompt,
            temperature=0.8,
            max_tokens=100,
            custom_param="test_value"
        )
        print(f"   ✅ Response: {response[:50]}...")

async def main():
    """Run all LLM tests"""
    print("🚀 Starting LLM Performance Testing")
    print("=" * 60)
    
    # Test 1: Basic simulation
    test_example_llm()
    
    # Test 2: Hugging Face style
    test_huggingface_llm()
    
    # Test 3: OpenAI style
    await test_openai_llm()
    
    # Test 4: Custom wrapper
    test_custom_llm()
    
    # Close monitor
    await monitor.close()
    
    print("\n" + "=" * 60)
    print("🎉 All LLM tests completed!")
    print("\n📊 View results:")
    print("   Dashboard: http://localhost:8080")
    print("   API docs:  http://localhost:8000/docs")
    print("   Health:    http://localhost:8000/health")

if __name__ == "__main__":
    asyncio.run(main()) 