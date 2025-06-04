#!/usr/bin/env python3
"""
Simple example: How to test your LLM with monitoring
"""

import time
import random
from monitoring.client import LLMMonitor

# Initialize monitor
monitor = LLMMonitor("http://localhost:8000")

def test_your_llm():
    """Simple example of LLM testing with monitoring"""
    
    print("🧠 Testing Your LLM with Monitoring")
    print("=" * 40)
    
    test_prompts = [
        "What is Python?",
        "Explain AI briefly",
        "How does a computer work?",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        
        # This is the key part - wrap your LLM call
        with monitor.track_request(model_name="my-llm") as tracker:
            
            # 1. Set prompt information
            prompt_tokens = len(prompt.split()) + 2  # Simple count
            tracker.set_prompt_info(
                tokens=prompt_tokens,
                length=len(prompt),
                temperature=0.7
            )
            
            # 2. Mark when processing starts
            tracker.start_processing()
            
            # 3. YOUR LLM CALL GOES HERE
            #    Replace this simulation with your actual LLM
            processing_time = random.uniform(0.5, 2.0)
            time.sleep(processing_time)  # Simulate LLM processing
            
            # Your LLM response (replace this)
            response = f"AI response to: {prompt}"
            
            # 4. Log the response
            completion_tokens = len(response.split()) + 1
            tracker.set_response_info(
                tokens=completion_tokens,
                length=len(response)
            )
            
            print(f"   ✅ Response: {response}")
        
        # The tracker automatically logs everything when exiting the context
    
    print(f"\n🎉 Completed {len(test_prompts)} tests!")
    print("📊 View results at: http://localhost:8080")

if __name__ == "__main__":
    test_your_llm() 