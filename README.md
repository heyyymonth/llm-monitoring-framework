# Minimalist LLM Performance Monitor

A lightweight, focused framework for monitoring LLM inference performance without fabricated data.

## Features

- **Real-time System Monitoring**: CPU, memory, and process metrics
- **LLM Inference Tracking**: Response times, token throughput, memory usage
- **Performance Dashboard**: Web-based visualization on port 8080
- **REST API**: Metrics collection on port 8000
- **WebSocket Updates**: Real-time metric streaming

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start monitoring**:
```bash
python main.py
```

3. **Access dashboard**: http://localhost:8080
4. **API docs**: http://localhost:8000/docs

## Usage

### Track LLM Inference

```python
import requests
from datetime import datetime

# Send inference metrics
response = requests.post("http://localhost:8000/track/inference", json={
    "request_id": "req-123",
    "model_name": "your-model",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "response_time_ms": 500.0,
    "tokens_per_second": 300.0,
    "memory_peak_mb": 256.0,
    "success": True,
    "timestamp": datetime.utcnow().isoformat()
})
```

### Get Current Metrics

```bash
curl http://localhost:8000/metrics/current
```

### Get Performance Summary

```bash
curl http://localhost:8000/metrics/summary?time_period=1h
```

## Testing

Run tests locally:
```bash
python tests/test_llm_metrics.py
python tests/test_monitoring.py
```

Run all tests with pytest:
```bash
python -m pytest tests/ -v
```

## API Endpoints

- `GET /health` - Service health status
- `POST /track/inference` - Log inference metrics
- `GET /metrics/current` - Current system & performance metrics
- `GET /metrics/history` - Historical metrics
- `GET /metrics/summary` - Performance summary
- `GET /stats` - Basic statistics
- `WS /ws/metrics` - Real-time metrics stream

## Project Structure

```
├── api/
│   └── server.py          # FastAPI server
├── monitoring/
│   ├── metrics.py         # Metrics collection
│   └── models.py          # Data models
├── dashboard/
│   └── app.py            # Dash dashboard
├── tests/                # Test suite
└── main.py              # Application entry point
```

## Real Data Only

This framework captures real LLM inference data without fabrication:
- Actual system resource usage
- Real inference response times  
- Genuine memory consumption
- Authentic token throughput metrics

## License

MIT License
