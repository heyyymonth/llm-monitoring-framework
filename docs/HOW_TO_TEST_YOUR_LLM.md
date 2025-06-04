# How to Test Your LLM with Performance Monitoring

## 🚀 **Quick Start (3 Steps)**

### Step 1: Start the Monitoring System
```bash
# Start the monitoring server
python main.py
```
This starts both the API server (port 8000) and dashboard (port 8080).

### Step 2: Wrap Your LLM Calls
```python
from monitoring.client import LLMMonitor

# Create monitor
monitor = LLMMonitor("http://localhost:8000")

# Wrap your LLM call
with monitor.track_request(model_name="your-model-name") as tracker:
    # Your LLM code here
    response = your_llm.generate("Your prompt")
    
    # Log the results
    tracker.set_response_info(tokens=100, length=len(response))
```

### Step 3: View Results
- **Dashboard**: http://localhost:8080
- **API**: http://localhost:8000/docs

---

## 🧠 **For Different LLM Types**

### 📱 **Local Hugging Face Model**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from monitoring.client import LLMMonitor

# Load your model
tokenizer = AutoTokenizer.from_pretrained("your-model")
model = AutoModelForCausalLM.from_pretrained("your-model")
monitor = LLMMonitor()

def generate_with_monitoring(prompt):
    with monitor.track_request(model_name="your-model") as tracker:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]
        
        tracker.set_prompt_info(
            tokens=prompt_tokens,
            length=len(prompt)
        )
        
        tracker.start_processing()
        
        # Generate
        outputs = model.generate(inputs.input_ids, max_length=100)
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion_tokens = len(outputs[0]) - prompt_tokens
        
        tracker.set_response_info(
            tokens=completion_tokens,
            length=len(response)
        )
        
        return response

# Test it
response = generate_with_monitoring("Tell me about AI")
```

### 🌐 **OpenAI API**
```python
import openai
from monitoring.client import LLMMonitor

openai.api_key = "your-api-key"
monitor = LLMMonitor()

def openai_with_monitoring(prompt):
    with monitor.track_request(model_name="gpt-3.5-turbo") as tracker:
        tracker.set_prompt_info(
            tokens=len(prompt.split()) * 1.3,  # Estimate
            length=len(prompt),
            temperature=0.7
        )
        
        tracker.start_processing()
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        tracker.set_response_info(
            tokens=response.usage.completion_tokens,
            length=len(response.choices[0].message.content)
        )
        
        return response.choices[0].message.content

# Test it
response = openai_with_monitoring("Explain quantum computing")
```

### 🔧 **Custom LLM/API**
```python
from monitoring.client import LLMMonitor

monitor = LLMMonitor()

def my_custom_llm_with_monitoring(prompt):
    with monitor.track_request(model_name="my-custom-llm") as tracker:
        # Set prompt info
        prompt_tokens = estimate_tokens(prompt)
        tracker.set_prompt_info(
            tokens=prompt_tokens,
            length=len(prompt),
            temperature=0.8
        )
        
        # Mark start of processing
        tracker.start_processing()
        
        # YOUR LLM CALL HERE
        response = your_llm_function(prompt)
        
        # Log response
        completion_tokens = estimate_tokens(response)
        tracker.set_response_info(
            tokens=completion_tokens,
            length=len(response)
        )
        
        # Add metadata
        tracker.set_metadata(
            model_version="1.0",
            custom_field="value"
        )
        
        return response

def estimate_tokens(text):
    """Simple token estimation - replace with your tokenizer"""
    return int(len(text.split()) * 1.3)
```

### 🏗️ **FastAPI LLM Service**
```python
from fastapi import FastAPI
from monitoring.client import LLMMonitor

app = FastAPI()
monitor = LLMMonitor()

@app.post("/generate")
async def generate_text(prompt: str):
    with monitor.track_request(model_name="api-llm") as tracker:
        # Set prompt info
        tracker.set_prompt_info(
            tokens=len(prompt.split()),
            length=len(prompt)
        )
        
        tracker.start_processing()
        
        # Your LLM processing
        response = await your_llm_generate(prompt)
        
        # Log response
        tracker.set_response_info(
            tokens=len(response.split()),
            length=len(response)
        )
        
        return {"response": response}
```

---

## 📊 **What Gets Monitored**

### ✅ **Automatic Metrics**
- **Response Time**: How long each request takes
- **Token Counts**: Input and output tokens
- **Throughput**: Requests per second
- **Error Rate**: Failed vs successful requests
- **System Resources**: CPU, Memory, GPU usage

### ✅ **Custom Metrics**
```python
# Add custom metadata
tracker.set_metadata(
    user_id="user123",
    prompt_category="coding",
    model_version="v2.1",
    custom_field="any_value"
)

# Send custom metrics
await monitor.send_custom_metric(
    "response_quality", 
    8.5, 
    {"evaluator": "human"}
)
```

---

## 🎯 **Testing Scenarios**

### **Performance Testing**
```python
# Test with varying loads
import asyncio

async def load_test():
    tasks = []
    for i in range(100):  # 100 concurrent requests
        task = asyncio.create_task(test_single_request(f"Prompt {i}"))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

# Monitor will track all requests automatically
```

### **Error Testing**
```python
# Test error handling
with monitor.track_request("error-test-model") as tracker:
    try:
        # Simulate error condition
        raise Exception("Simulated LLM error")
    except Exception as e:
        # Error is automatically logged
        pass  # Tracker handles error logging
```

### **A/B Testing**
```python
# Compare different models
models = ["model-a", "model-b", "model-c"]

for model in models:
    with monitor.track_request(model_name=model) as tracker:
        response = test_model(model, "Same prompt for all")
        # Each model's performance is tracked separately
```

---

## 🎨 **Dashboard Features**

Visit http://localhost:8080 to see:

- **📈 Real-time Charts**: Response times, throughput, error rates
- **🏆 Performance Metrics**: P95, P99 response times, tokens/sec
- **🚨 Active Alerts**: When thresholds are exceeded
- **🤖 Model Comparison**: Performance across different models
- **📊 System Resources**: CPU, Memory, GPU usage

---

## 🔧 **Run Your Test**

1. **Start monitoring**: `python main.py`
2. **Run your test**: `python test_my_llm.py`
3. **View results**: Open http://localhost:8080

The framework automatically handles all the monitoring, storage, and visualization! 