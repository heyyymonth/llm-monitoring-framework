# Examples

This directory contains example code and integration examples for the LLM monitoring framework.

## Directory Structure

- `integrations/` - Real-world integration examples showing how to use the framework with various LLM providers

## Available Examples

### Integrations

- `ollama_integration_example.py` - Shows how to integrate Ollama LLM requests with automatic monitoring

## Running Examples

Make sure the monitoring framework is running first:

```bash
# Start the monitoring framework
python main.py

# In another terminal, run examples
python examples/integrations/ollama_integration_example.py
```

## Requirements

- LLM monitoring framework running (port 8000)
- Dashboard running (port 8080) 
- Respective LLM providers (e.g., Ollama for ollama examples) 