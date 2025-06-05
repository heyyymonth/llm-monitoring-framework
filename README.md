# LLM Performance Monitor

Minimal, focused monitoring system for LLM inference performance and system health.

## Features

🧠 **LLM-Specific Metrics**
- Inference response times and token throughput
- GPU utilization and memory usage
- Queue metrics and request processing
- Model loading performance
- Cache hit rates and memory pressure

🔥 **Enhanced System-Level Monitoring**
- Disk I/O performance (IOPS, latency, queue depth)
- Network interface monitoring (18+ interfaces tracked)
- Memory fragmentation and swap pressure detection
- CPU temperature and thermal throttling detection
- Process-specific resource tracking and thread analysis

📊 **Real-Time Dashboard**
- Simple, clean interface focused on LLM performance
- Key performance indicators and health scoring
- Performance recommendations

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Start the API Server**
```bash
python api/server.py
```

3. **Start the Dashboard**
```bash
python dashboard/app.py
```

4. **View Dashboard**
Open http://localhost:8080 in your browser

## Testing

Run the LLM metrics tests:
```bash
python tests/test_llm_metrics.py
python tests/test_focused_metrics.py
python tests/test_enhanced_system_metrics.py
```

Run all tests with pytest:
```bash
python -m pytest tests/ -v
```

## API Endpoints

- `GET /health` - System health and LLM metrics
- `GET /metrics/current` - Current performance metrics
- `POST /inference` - Log inference metrics
- `POST /error` - Log error metrics

## Configuration

Key metrics monitored:

### Critical for LLM Performance
- **GPU Utilization**: Optimal range 40-90%
- **Memory Pressure**: Alerts when <2GB available
- **Disk I/O**: Read rates for model loading performance
- **Memory Fragmentation**: Impact on large model loading
- **Thermal Throttling**: Performance impact detection
- **Queue Times**: Target <300ms average
- **Response Times**: Target <2000ms average

### Health Scoring
- 80-100: Excellent LLM performance
- 60-79: Good performance
- 40-59: Fair with some issues
- 0-39: Poor performance requiring attention

## Architecture

```
├── monitoring/
│   ├── models.py      # LLM-focused data models
│   ├── metrics.py     # Metrics collection
│   └── database.py    # Storage
├── api/
│   └── server.py      # FastAPI server
├── dashboard/
│   └── app.py         # Dash dashboard
└── test_llm_metrics.py # Testing
```

## Alerts

Automatic alerts for:
- Memory pressure (>85% usage)
- Thermal throttling events
- High error rates (>1%)
- Poor GPU utilization (<40% or >95%)
- High queue times (>500ms)

## Documentation

📋 **[Validation Summary](docs/VALIDATION_SUMMARY.md)** - Comprehensive testing results with real Ollama inference

## License

MIT License
