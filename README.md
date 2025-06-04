# 🚀 LLM Performance Monitoring Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Dash](https://img.shields.io/badge/Dash-2.14.1-purple.svg)](https://dash.plotly.com/)
[![Coverage](https://img.shields.io/badge/coverage-77%25-brightgreen.svg)](https://github.com/heyyymonth/llm-monitoring-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive, production-ready monitoring framework for Large Language Models with real-time performance tracking, interactive dashboards, and intelligent alerting.**

![Dashboard Preview](screenshots/dashboard-preview.png)

## 🌟 **Why Choose This Framework?**

- **🔥 Production-Ready**: Battle-tested with real LLM workloads
- **📊 Real-Time Monitoring**: Live dashboards with interactive charts
- **🚨 Intelligent Alerts**: Configurable thresholds and notifications
- **🔌 Multi-LLM Support**: OpenAI, Hugging Face, Ollama, custom APIs
- **⚡ Easy Integration**: 5-minute setup with comprehensive SDK
- **🧪 Thoroughly Tested**: 33 tests with 77% coverage

---

## 🚀 **Quick Start**

### **1. Installation**

```bash
git clone https://github.com/heyyymonth/llm-monitoring-framework.git
cd llm-monitoring-framework
pip install -r requirements.txt
```

### **2. Start the Framework**

```bash
python main.py
```

**Access Points:**
- 📊 **Dashboard**: http://localhost:8080 (Real-time monitoring)
- 🔗 **API**: http://localhost:8000 (RESTful endpoints)
- 📚 **API Docs**: http://localhost:8000/docs (Interactive documentation)

### **3. Test Integration**

```bash
python examples/integrations/quick_test.py
```

---

## 💻 **Usage Examples**

### **OpenAI Integration**

```python
from monitoring.client import LLMMonitor
import openai

monitor = LLMMonitor()

with monitor.track_request(model_name="gpt-4") as tracker:
    # Your OpenAI call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Automatically tracked: tokens, latency, success/failure
```

### **Hugging Face Integration**

```python
from transformers import pipeline
from monitoring.client import LLMMonitor

monitor = LLMMonitor()
generator = pipeline('text-generation', model='gpt2')

with monitor.track_request(model_name="gpt2") as tracker:
    result = generator("The future of AI is", max_length=50)
    # Performance metrics automatically captured
```

### **Ollama Integration**

```python
import requests
from monitoring.client import LLMMonitor

monitor = LLMMonitor()

with monitor.track_request(model_name="stable-code") as tracker:
    response = requests.post('http://localhost:11434/api/generate', 
                           json={"model": "stable-code", "prompt": "Write a function"})
    # Real-time performance tracking
```

### **Custom API Integration**

```python
from monitoring.client import LLMMonitor

monitor = LLMMonitor()

def call_my_llm(prompt):
    with monitor.track_request(model_name="my-custom-llm") as tracker:
        tracker.set_prompt_info(tokens=len(prompt.split()), length=len(prompt))
        tracker.start_processing()
        
        # Your LLM API call here
        response = my_llm_api.generate(prompt)
        
        tracker.set_response_info(tokens=len(response.split()), length=len(response))
        return response
```

---

## 📊 **Features**

### **🎯 Core Monitoring**
- **Performance Metrics**: Response time, throughput, token counts
- **System Monitoring**: CPU, memory, GPU utilization
- **Error Tracking**: Failure rates, error messages, stack traces
- **Request Correlation**: End-to-end request tracing

### **📈 Real-Time Dashboard**
- **Interactive Charts**: Live performance visualizations
- **Multi-Model View**: Compare performance across different LLMs
- **Alert Status**: Real-time alert monitoring
- **Historical Data**: Trend analysis and reporting

### **🔔 Intelligent Alerting**
- **Configurable Thresholds**: Custom alert rules
- **Multiple Channels**: Email, Slack, webhook notifications
- **Smart Filtering**: Reduce alert noise with intelligent grouping
- **Escalation Policies**: Multi-tier alert management

### **🔌 Easy Integration**
- **SDK Support**: Python client with context managers
- **REST API**: Full RESTful API for custom integrations
- **Multi-LLM**: Support for all major LLM providers
- **Custom Metrics**: Extensible for domain-specific monitoring

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Your LLM App  │───▶│  Monitoring SDK │───▶│   FastAPI API   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Dash Dashboard │◀───│   Database      │
                       │  (Port 8080)    │    │  (SQLite/Redis) │
                       └─────────────────┘    └─────────────────┘
```

---

## ⚙️ **Configuration**

### **Environment Variables**
```bash
export LLM_MONITOR_API_HOST=localhost
export LLM_MONITOR_API_PORT=8000
export LLM_MONITOR_DASHBOARD_PORT=8080
export LLM_MONITOR_DATABASE_URL=sqlite:///data/monitoring.db
```

### **Configuration File (`config.yaml`)**
```yaml
monitoring:
  collection_interval: 5  # seconds
  retention_days: 30
  
alerts:
  response_time_threshold: 5000  # ms
  error_rate_threshold: 0.05     # 5%
  
dashboard:
  refresh_interval: 2  # seconds
  max_data_points: 1000
```

---

## 🧪 **Testing**

### **Run All Tests**
```bash
python tests/run_tests.py
```

**Test Results:**
- ✅ **33/33 tests passing**
- ✅ **77% code coverage**
- ✅ **Real LLM integration tested**

### **Test Categories**
- **Unit Tests**: Core functionality validation
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **LLM Integration**: Real model testing with Ollama

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Quick Development Setup**
```bash
git clone https://github.com/heyyymonth/llm-monitoring-framework.git
cd llm-monitoring-framework
pip install -r requirements.txt
pip install -e .  # Editable install
```

### **Running Tests**
```bash
pytest tests/ -v --cov=monitoring
```

---

## 📚 **Documentation**

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed installation instructions
- **[Features](docs/FEATURES.md)** - Comprehensive feature documentation
- **[API Reference](http://localhost:8000/docs)** - Interactive API documentation
- **[Integration Guide](docs/HOW_TO_TEST_YOUR_LLM.md)** - LLM integration examples
- **[Contributing](CONTRIBUTING.md)** - Development guidelines

---

## 📈 **Roadmap**

### **v1.1 (Next Release)**
- [ ] **Distributed Tracing**: OpenTelemetry integration
- [ ] **Advanced Analytics**: ML-powered performance insights
- [ ] **Custom Dashboards**: User-defined dashboard layouts
- [ ] **Export Features**: Data export to external systems

### **v1.2 (Future)**
- [ ] **Multi-Tenant Support**: Organization-level isolation
- [ ] **Advanced Alerting**: Machine learning-based anomaly detection
- [ ] **Performance Optimization**: Caching and query optimization
- [ ] **Cloud Deployment**: Docker and Kubernetes support

---

## 🐛 **Support**

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/heyyymonth/llm-monitoring-framework/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/heyyymonth/llm-monitoring-framework/discussions)
- **📧 Questions**: [Create an Issue](https://github.com/heyyymonth/llm-monitoring-framework/issues/new)

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

Built with ❤️ using:
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance API framework
- **[Dash](https://dash.plotly.com/)** - Interactive web applications
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation and settings management
- **[SQLite](https://sqlite.org/)** - Lightweight database
- **[Redis](https://redis.io/)** - In-memory data structure store

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[🚀 Get Started](#-quick-start) • [📚 Docs](docs/) • [🤝 Contribute](CONTRIBUTING.md) • [🐛 Issues](https://github.com/heyyymonth/llm-monitoring-framework/issues)

</div> 