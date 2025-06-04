# 🛠️ **Complete Setup Guide**

## 📋 **System Requirements**

### **Minimum Requirements**
- **Python**: 3.8 or higher
- **RAM**: 2GB available memory
- **Storage**: 1GB free disk space
- **Network**: Internet access for package installation

### **Recommended Requirements**
- **Python**: 3.9+ 
- **RAM**: 4GB+ available memory
- **Storage**: 5GB+ free disk space (for historical data)
- **CPU**: 2+ cores
- **GPU**: NVIDIA GPU with CUDA drivers (optional, for GPU monitoring)

### **Supported Operating Systems**
✅ **Linux** (Ubuntu 18.04+, CentOS 7+, RHEL 7+)  
✅ **macOS** (10.14+)  
✅ **Windows** (10, 11)  

---

## 🚀 **Installation Methods**

### **Method 1: Quick Install (Recommended)**

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-monitoring-framework.git
cd llm-monitoring-framework

# Run the automated installer
chmod +x install.sh
./install.sh

# Start monitoring
python main.py
```

### **Method 2: Manual Installation**

```bash
# 1. Clone repository
git clone https://github.com/yourusername/llm-monitoring-framework.git
cd llm-monitoring-framework

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify installation
python quick_test.py

# 6. Start services
python main.py
```

### **Method 3: Docker Installation** (Coming Soon)

```bash
# Using Docker Compose
docker-compose up -d

# Using Docker directly
docker build -t llm-monitor .
docker run -p 8000:8000 -p 8080:8080 llm-monitor
```

---

## 🔧 **Configuration Setup**

### **Basic Configuration**
Edit `config.yaml` for your environment:

```yaml
# config.yaml
monitoring:
  metrics_interval: 1.0        # Metrics collection frequency (seconds)
  max_history_days: 30         # Data retention period
  alert_thresholds:
    cpu_percent: 80            # CPU alert threshold
    memory_percent: 85         # Memory alert threshold  
    response_time_ms: 5000     # Response time alert (ms)
    error_rate_percent: 10     # Error rate alert

api:
  host: "0.0.0.0"             # API server host
  port: 8000                  # API server port
  workers: 1                  # Number of worker processes
  log_level: "info"           # Logging level

dashboard:
  host: "0.0.0.0"             # Dashboard host
  port: 8080                  # Dashboard port
  debug: false                # Debug mode
  update_interval: 1000       # Update frequency (ms)

database:
  sqlite_path: "data/monitoring.db"  # SQLite database path
  redis_host: "localhost"            # Redis server host
  redis_port: 6379                   # Redis server port
  redis_db: 0                        # Redis database number
```

### **Advanced Configuration**

#### **Production Settings**
```yaml
# For production environments
monitoring:
  metrics_interval: 0.5      # More frequent collection
  max_history_days: 90       # Longer retention

api:
  workers: 4                 # Multiple workers for high load
  log_level: "warning"       # Reduce log verbosity

database:
  redis_host: "redis-server" # External Redis instance
  redis_port: 6379
  redis_password: "your-password"  # Redis authentication
```

#### **Development Settings**
```yaml
# For development/testing
monitoring:
  metrics_interval: 2.0      # Less frequent collection
  max_history_days: 7        # Shorter retention

dashboard:
  debug: true                # Enable debug mode
  
api:
  log_level: "debug"         # Verbose logging
```

---

## 🐳 **Environment-Specific Setup**

### **Linux (Ubuntu/Debian)**

```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv redis-server

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Install NVIDIA monitoring (optional)
sudo apt install nvidia-ml-py3

# Clone and setup
git clone https://github.com/yourusername/llm-monitoring-framework.git
cd llm-monitoring-framework
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start monitoring
python main.py
```

### **macOS**

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python redis

# Start Redis service
brew services start redis

# Clone and setup
git clone https://github.com/yourusername/llm-monitoring-framework.git
cd llm-monitoring-framework
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start monitoring
python main.py
```

### **Windows**

```powershell
# Install Python (from python.org)
# Install Git (from git-scm.com)

# Install Redis for Windows
# Download from: https://github.com/MicrosoftArchive/redis/releases

# Clone repository
git clone https://github.com/yourusername/llm-monitoring-framework.git
cd llm-monitoring-framework

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start monitoring
python main.py
```

---

## 🔌 **LLM Integration Setup**

### **OpenAI Integration**

```python
# install_openai.py
import openai
from monitoring.client import LLMMonitor

# Setup monitoring
monitor = LLMMonitor("http://localhost:8000")
client = openai.OpenAI(api_key="your-api-key")

def monitored_completion(prompt, **kwargs):
    with monitor.track_request(model_name="gpt-3.5-turbo") as tracker:
        # Set prompt info
        tracker.set_prompt_info(
            tokens=len(prompt.split()),  # Rough estimate
            length=len(prompt),
            temperature=kwargs.get('temperature', 0.7)
        )
        
        tracker.start_processing()
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        # Set response info
        tracker.set_response_info(
            tokens=response.usage.completion_tokens,
            length=len(response.choices[0].message.content)
        )
        
        return response.choices[0].message.content

# Test the integration
if __name__ == "__main__":
    result = monitored_completion("Hello, how are you?")
    print(f"Response: {result}")
```

### **Hugging Face Integration**

```python
# install_huggingface.py
from transformers import pipeline, AutoTokenizer
from monitoring.client import LLMMonitor

# Setup
monitor = LLMMonitor("http://localhost:8000")
model_name = "gpt2"
generator = pipeline("text-generation", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def monitored_generation(prompt, **kwargs):
    with monitor.track_request(model_name=model_name) as tracker:
        # Tokenize for accurate counts
        prompt_tokens = tokenizer.encode(prompt)
        
        tracker.set_prompt_info(
            tokens=len(prompt_tokens),
            length=len(prompt),
            max_tokens=kwargs.get('max_length', 100)
        )
        
        tracker.start_processing()
        
        # Generate
        outputs = generator(prompt, **kwargs)
        response = outputs[0]['generated_text']
        
        # Calculate response tokens
        response_only = response[len(prompt):]
        response_tokens = tokenizer.encode(response_only)
        
        tracker.set_response_info(
            tokens=len(response_tokens),
            length=len(response_only)
        )
        
        return response

# Test the integration
if __name__ == "__main__":
    result = monitored_generation("The future of AI is", max_length=50)
    print(f"Generated: {result}")
```

### **Ollama Integration**

```python
# install_ollama.py
import requests
import json
from monitoring.client import LLMMonitor

class OllamaMonitor:
    def __init__(self, base_url="http://localhost:11434", monitor_url="http://localhost:8000"):
        self.base_url = base_url
        self.monitor = LLMMonitor(monitor_url)
    
    def generate(self, model, prompt, **kwargs):
        with self.monitor.track_request(model_name=model) as tracker:
            # Estimate prompt tokens (rough)
            prompt_tokens = len(prompt.split()) * 1.3  # Approximation
            
            tracker.set_prompt_info(
                tokens=int(prompt_tokens),
                length=len(prompt),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            tracker.start_processing()
            
            # Make Ollama API call
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
            )
            
            result = response.json()
            
            # Use Ollama's token counts if available
            completion_tokens = result.get('eval_count', 0)
            response_text = result.get('response', '')
            
            tracker.set_response_info(
                tokens=completion_tokens,
                length=len(response_text)
            )
            
            # Add Ollama-specific metadata
            tracker.set_metadata(
                eval_duration=result.get('eval_duration', 0),
                load_duration=result.get('load_duration', 0),
                prompt_eval_count=result.get('prompt_eval_count', 0),
                ollama_model=model
            )
            
            return response_text

# Test the integration
if __name__ == "__main__":
    ollama = OllamaMonitor()
    response = ollama.generate("llama2", "Explain quantum computing in simple terms")
    print(f"Response: {response}")
```

---

## 🚦 **Service Management**

### **Starting Services**

#### **Option 1: All-in-One (Recommended)**
```bash
python main.py
```

#### **Option 2: Individual Services**
```bash
# Terminal 1: API Server
python -m api.server

# Terminal 2: Dashboard
python -m dashboard.app

# Terminal 3: Test the setup
python quick_test.py
```

#### **Option 3: Background Services**
```bash
# Start API server in background
nohup python -m api.server > logs/api.log 2>&1 &

# Start dashboard in background
nohup python -m dashboard.app > logs/dashboard.log 2>&1 &

# Check status
curl http://localhost:8000/health
```

### **Service Status Verification**

```bash
# Check all services
python -c "
import requests
import sys

try:
    # Test API
    api_health = requests.get('http://localhost:8000/health').json()
    print(f'✅ API Server: {api_health[\"status\"]}')
    
    # Test Dashboard
    dashboard = requests.get('http://localhost:8080')
    print(f'✅ Dashboard: {\"Healthy\" if dashboard.status_code == 200 else \"Error\"}')
    
    print('🎉 All services running successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"
```

### **Stopping Services**

```bash
# If running with main.py (Ctrl+C)
# Or kill background processes
pkill -f "python -m api.server"
pkill -f "python -m dashboard.app"
pkill -f "python main.py"
```

---

## 🧪 **Testing Your Setup**

### **Basic Functionality Test**
```bash
python quick_test.py
```

### **Full Test Suite**
```bash
python run_tests.py
```

### **Integration Test with Real LLM**
```bash
# Test with Ollama (if installed)
python test_ollama_llm.py

# Test with custom LLM
python test_my_llm.py
```

### **Dashboard Test**
1. Open http://localhost:8080
2. Verify real-time charts update
3. Check that system metrics are displayed
4. Test different time ranges

### **API Test**
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics/current
curl http://localhost:8000/docs  # Interactive API documentation
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Redis Connection Error**
```bash
# Check if Redis is running
redis-cli ping

# Start Redis if not running
# Linux/macOS: sudo systemctl start redis
# Windows: Start Redis from installation directory
```

#### **Port Already in Use**
```bash
# Check what's using the ports
lsof -i :8000  # API port
lsof -i :8080  # Dashboard port

# Kill processes or change ports in config.yaml
```

#### **Python Module Not Found**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### **GPU Monitoring Not Working**
```bash
# Install NVIDIA drivers and tools
nvidia-smi  # Should show GPU information

# Install Python GPU monitoring
pip install nvidia-ml-py3
```

### **Getting Help**

1. **Check Logs**: Look in `logs/` directory for error messages
2. **Run Diagnostics**: `python quick_test.py` for basic health check
3. **Test Individual Components**: Use the test files in the repository
4. **GitHub Issues**: Report bugs at [repository issues page]
5. **Documentation**: Check `FEATURES.md` for detailed capabilities

---

## 🔄 **Next Steps**

After successful setup:

1. **✅ Integrate with your LLM** - Use the examples above
2. **📊 Explore the Dashboard** - http://localhost:8080
3. **📖 Read the API Docs** - http://localhost:8000/docs
4. **🧪 Run Test Suite** - `python run_tests.py`
5. **📝 Customize Configuration** - Edit `config.yaml`
6. **🚨 Set Up Alerts** - Configure alert thresholds
7. **📈 Monitor Performance** - Watch your LLM metrics in real-time

**🎉 Congratulations! Your LLM monitoring framework is ready for production use.** 