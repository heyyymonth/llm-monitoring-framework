# 🧠 LLM Performance Monitoring Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-33%2F33%20✅-brightgreen.svg)](./TEST_RESULTS_SUMMARY.md)
[![Coverage](https://img.shields.io/badge/Coverage-77%25-green.svg)](./htmlcov/index.html)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

> **Enterprise-grade monitoring for Large Language Models with real-time performance tracking, comprehensive analytics, and production-ready dashboards.**

---

## 🚀 **Why Use This Framework?**

### **🎯 Production-Ready LLM Monitoring**
✅ **Real-time Performance Tracking** - Monitor response times, throughput, token generation  
✅ **Resource Utilization** - Track CPU, memory, GPU usage during inference  
✅ **Error Analytics** - Comprehensive error tracking and root cause analysis  
✅ **Scale Management** - Handle thousands of requests with queue management  
✅ **Cost Optimization** - Token usage analytics for budget control  

### **📊 Enterprise Features**
✅ **Real-time Dashboard** - Beautiful web interface with live charts  
✅ **REST API** - Full programmatic access to all metrics  
✅ **Alert System** - Configurable thresholds with notifications  
✅ **Data Export** - Export metrics for compliance and reporting  
✅ **Multi-LLM Support** - Monitor multiple models simultaneously  

### **🔧 Developer Experience**
✅ **5-Minute Setup** - Get monitoring running in minutes  
✅ **Framework Agnostic** - Works with any LLM (OpenAI, Hugging Face, Ollama, custom)  
✅ **Comprehensive Tests** - 33 tests with 77% coverage  
✅ **Production Tested** - Battle-tested with real LLM workloads  

---

## 📋 **Table of Contents**

- [Quick Start](#-quick-start)
- [Installation](#-installation) 
- [Usage Examples](#-usage-examples)
- [Dashboard Screenshots](#-dashboard-screenshots)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)

---

## 🏃 **Quick Start**

### **1. Clone & Install**
```bash
git clone https://github.com/yourusername/llm-monitoring-framework.git
cd llm-monitoring-framework
pip install -r requirements.txt
```

### **2. Start Monitoring Services**
```bash
# Option 1: Start everything (recommended)
python main.py

# Option 2: Start components separately
python -m api.server &     # API server on :8000
python -m dashboard.app &  # Dashboard on :8080
```

### **3. Access Your Dashboard**
🌐 **Dashboard**: http://localhost:8080  
📖 **API Docs**: http://localhost:8000/docs  
❤️ **Health Check**: http://localhost:8000/health  

### **4. Integrate with Your LLM** (30 seconds)
```python
from monitoring.client import LLMMonitor

monitor = LLMMonitor("http://localhost:8000")

# Wrap any LLM call
with monitor.track_request(model_name="your-model") as tracker:
    tracker.set_prompt_info(tokens=len(prompt_tokens), length=len(prompt))
    tracker.start_processing()
    
    # Your LLM inference here
    response = your_llm_model.generate(prompt)
    
    tracker.set_response_info(tokens=len(response_tokens), length=len(response))
```

**🎉 That's it! Your LLM is now monitored with real-time metrics.**

---

## 💻 **Installation**

### **Prerequisites**
- Python 3.8+
- 4GB+ RAM (for dashboard + monitoring)
- Optional: NVIDIA GPU with drivers (for GPU monitoring)

### **Installation Options**

#### **Option 1: Standard Installation**
```bash
pip install -r requirements.txt
```

#### **Option 2: Development Installation**  
```bash
pip install -e .
pip install -r requirements-dev.txt  # Additional dev tools
```

#### **Option 3: Docker Installation** (Coming Soon)
```bash
docker-compose up -d
```

### **Verify Installation**
```bash
python quick_test.py  # Basic functionality test
python run_tests.py  # Full test suite (33 tests)
```

---

## 🔥 **Usage Examples**

### **OpenAI Integration**
```python
import openai
from monitoring.client import LLMMonitor

monitor = LLMMonitor()
client = openai.OpenAI()

with monitor.track_request(model_name="gpt-3.5-turbo") as tracker:
    tracker.set_prompt_info(tokens=15, length=100, temperature=0.7)
    tracker.start_processing()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    tracker.set_response_info(
        tokens=response.usage.completion_tokens,
        length=len(response.choices[0].message.content)
    )
```

### **Hugging Face Integration**
```python
from transformers import pipeline
from monitoring.client import LLMMonitor

monitor = LLMMonitor()
generator = pipeline("text-generation", model="gpt2")

with monitor.track_request(model_name="gpt2") as tracker:
    tracker.set_prompt_info(tokens=10, length=50)
    tracker.start_processing()
    
    outputs = generator("Hello world", max_length=100)
    
    tracker.set_response_info(tokens=25, length=200)
```

### **Ollama Integration**
```python
import requests
from monitoring.client import LLMMonitor

monitor = LLMMonitor()

with monitor.track_request(model_name="llama2") as tracker:
    tracker.set_prompt_info(tokens=12, length=80)
    tracker.start_processing()
    
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama2',
        'prompt': 'Explain quantum computing'
    })
    
    result = response.json()
    tracker.set_response_info(
        tokens=result.get('eval_count', 0),
        length=len(result.get('response', ''))
    )
```

### **Custom LLM Integration**
```python
from monitoring.client import LLMMonitor

class MyLLMWrapper:
    def __init__(self, model):
        self.model = model
        self.monitor = LLMMonitor()
    
    def generate(self, prompt, **kwargs):
        with self.monitor.track_request(model_name="my-model") as tracker:
            # Set prompt info
            tracker.set_prompt_info(
                tokens=self.count_tokens(prompt),
                length=len(prompt),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            tracker.start_processing()
            
            # Your model inference
            response = self.model.generate(prompt, **kwargs)
            
            # Set response info  
            tracker.set_response_info(
                tokens=self.count_tokens(response),
                length=len(response)
            )
            
            return response
```

---

## 📊 **Dashboard Screenshots**

### **Live Metrics**
![Live Metrics](screenshots/live_metrics.png)

### **Historical Performance**
![Historical Performance](screenshots/historical_performance.png)

### **Alert System**
![Alert System](screenshots/alert_system.png)

## �� **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Service   │    │  Monitor API    │    │   Dashboard     │
│                 │───▶│                 │───▶│                 │
│  (Your Model)   │    │  FastAPI Server │    │  Web Interface  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Data Storage   │              │
         └──────────────▶│                 │◀─────────────┘
                        │ SQLite + Redis  │
                        └─────────────────┘
```

## 🔧 **Configuration**

Edit `config.yaml` to customize monitoring settings:

```yaml
monitoring:
  metrics_interval: 1.0  # seconds
  max_history_days: 30
  alert_thresholds:
    cpu_percent: 80
    memory_percent: 85
    response_time_ms: 5000

dashboard:
  host: "0.0.0.0"
  port: 8080
  update_interval: 1000  # milliseconds

api:
  host: "0.0.0.0"
  port: 8000
```

## 📖 **API Reference**

- `POST /track/inference` - Log inference metrics
- `GET /metrics/current` - Current system metrics
- `GET /metrics/history` - Historical performance data
- `GET /health` - Service health check
- `WebSocket /ws/metrics` - Real-time metric stream

## 🧪 **Testing**

### **Unit Tests**
```bash
python run_tests.py
```

### **Coverage**
```bash
coverage run -m pytest
coverage html
```

## 🤝 **Contributing**

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 **License**

MIT License 