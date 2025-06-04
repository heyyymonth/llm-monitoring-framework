# LLM Performance Monitoring Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready monitoring framework for Large Language Models with real-time performance tracking and alerting.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start monitoring system
python main.py

# Access dashboard: http://localhost:8080
# API docs: http://localhost:8000/docs
```

## Usage

```python
from monitoring import LLMMonitor

# Initialize monitor
monitor = LLMMonitor()

# Track LLM inference
with monitor.track_request(model_name="gpt-4") as tracker:
    tracker.set_prompt_info(tokens=100, length=len(prompt))
    tracker.start_processing()
    
    # Your LLM call here
    response = llm.generate(prompt)
    
    tracker.set_response_info(tokens=150, length=len(response))
```

## Components

- **Monitoring**: Real-time metrics collection and storage
- **API**: RESTful endpoints for metrics and alerts
- **Dashboard**: Live web interface with charts and alerts
- **Database**: SQLite + Redis for persistent and cached data
- **Client SDK**: Easy integration with any LLM provider

## Features

- Real-time performance monitoring
- Multi-LLM provider support (OpenAI, Hugging Face, Ollama, Custom)
- Automatic alerting and threshold monitoring
- Resource utilization tracking (CPU, memory, GPU)
- Token usage and cost analytics
- Error tracking and debugging

## Testing

```bash
python tests/run_tests.py
```

## Configuration

Edit `config.yaml` for custom settings:
- Database paths
- Alert thresholds  
- API/Dashboard ports
- Monitoring intervals

## License

MIT
