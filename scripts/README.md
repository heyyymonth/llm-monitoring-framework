# Scripts

This directory contains utility scripts and test scripts for the LLM monitoring framework.

## Available Scripts

- `real_ollama_test.py` - Comprehensive test script that makes real Ollama requests and verifies automatic tracking

## Running Scripts

Make sure the monitoring framework is running first:

```bash
# Start the monitoring framework
python main.py

# In another terminal, run test scripts
python scripts/real_ollama_test.py
```

## Purpose

These scripts are useful for:
- Testing the framework with real LLM providers
- Validating automatic request tracking
- Performance testing
- Integration testing

## Requirements

- LLM monitoring framework running (port 8000)
- Dashboard running (port 8080)
- Respective LLM providers (e.g., Ollama for Ollama tests) 