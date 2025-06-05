# LLM Monitoring Framework - Ollama Integration Examples

This directory contains practical examples showing how to integrate Ollama with the LLM Monitoring Framework.

## Prerequisites

1. **Monitoring Framework Running**: Make sure your monitoring system is running:
   ```bash
   # From the main serve directory
   python main.py
   ```
   This starts:
   - API server at http://localhost:8000
   - Dashboard at http://localhost:8080

2. **Ollama Installed**: Make sure Ollama is installed and has models available:
   ```bash
   ollama list  # Check available models
   ```

## Examples

### 1. `single_ollama_call.py` - Quick Single Call

Simple example for making one monitored Ollama call.

**Usage:**
```bash
# With default prompt
python examples/single_ollama_call.py

# With custom prompt
python examples/single_ollama_call.py "Explain machine learning in simple terms"
```

### 2. `ollama_integration.py` - Multiple Calls Demo

Comprehensive example showing multiple Ollama calls with detailed monitoring.

**Usage:**
```bash
python examples/ollama_integration.py
```

This will run 3 example prompts and automatically track all metrics.

## How It Works

1. **Import Framework**: Examples import the monitoring client from the parent directory
2. **Create Monitor**: Use `create_monitor()` to connect to your running monitoring API
3. **Context Manager**: Use `with monitor.track_request()` to automatically track the call
4. **Set Metrics**: Provide prompt/response info for accurate tracking
5. **View Results**: Check the dashboard at http://localhost:8080

## What Gets Tracked

- **Response Time**: How long each call takes
- **Token Counts**: Estimated input/output tokens
- **Model Name**: Which Ollama model was used
- **Success/Failure**: Whether the call succeeded
- **Metadata**: Additional info like actual response times

## Viewing Results

After running examples:
1. Open http://localhost:8080 in your browser
2. See real-time metrics and performance data
3. Track trends across multiple calls

## Customization

You can modify the examples to:
- Use different Ollama models (`model="mistral-small3.1"`)
- Change monitoring URL (`monitor_url="http://different-host:8000"`)
- Add custom metadata
- Handle different types of prompts

## Troubleshooting

- **Import errors**: Make sure you're running from the main serve directory
- **Connection errors**: Ensure the monitoring API is running at localhost:8000
- **Ollama errors**: Check that Ollama is installed and models are available 