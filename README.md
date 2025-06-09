# LLM Quality & Safety Monitor

A production-ready framework for monitoring LLM applications with focus on quality, safety, and observability - not just system metrics.

## Why This Exists

Most LLM monitoring tools focus on system performance (CPU, memory) rather than what actually matters in production:
- **Quality Drift**: When your model starts giving worse answers
- **Safety Violations**: Hallucinations, bias, toxic outputs  
- **Cost Explosion**: Uncontrolled token usage
- **Compliance Gaps**: PII leakage, audit trail failures

This framework focuses on the **quality and safety** aspects that make or break real LLM applications.

## Core Features

### 🛡️ Safety & Quality Monitoring
- **Hallucination Detection**: Automated fact-checking and consistency analysis
- **Toxicity Filtering**: Real-time content safety evaluation
- **Bias Detection**: Automated fairness and bias assessment
- **PII Protection**: Detect and prevent sensitive data leakage

### 📊 LLM-Specific Observability  
- **Prompt Tracing**: Full request lifecycle visibility
- **Response Quality**: Semantic similarity, coherence, relevance
- **Cost Tracking**: Token usage, model costs, efficiency metrics
- **Drift Detection**: Monitor prompt and response pattern changes

### 🔄 Feedback & Optimization
- **Human-in-the-Loop**: Collect and analyze user feedback
- **A/B Testing**: Compare prompt variations and model versions
- **Quality Metrics**: Automated evaluation with custom criteria

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

### Monitor LLM Quality

```python
from monitoring.quality import QualityMonitor

monitor = QualityMonitor()

# Evaluate response quality
result = monitor.evaluate_response(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    check_hallucination=True,
    check_toxicity=True,
    check_bias=True
)

print(f"Quality Score: {result.quality_score}")
print(f"Safety Flags: {result.safety_flags}")
```

### Track Costs and Performance

```python
from monitoring.cost import CostTracker

tracker = CostTracker()

# Log inference with cost tracking
tracker.log_inference(
    model="gpt-4",
    prompt_tokens=100,
    completion_tokens=50,
    cost_per_token=0.00003
)

# Get cost analysis
analysis = tracker.get_cost_analysis(timeframe="24h")
```

## API Endpoints

- `POST /monitor/inference` - Log and evaluate LLM inference
- `GET /metrics/quality` - Quality metrics and trends
- `GET /metrics/safety` - Safety violations and patterns  
- `GET /metrics/cost` - Cost analysis and optimization
- `GET /feedback` - User feedback and ratings
- `POST /evaluate` - Batch evaluation of responses

## Real-World Focus

Unlike generic monitoring tools, this framework addresses actual production concerns:

### Enterprise Requirements
- **Compliance**: Built-in audit trails and regulatory reporting
- **Security**: Prompt injection detection and PII protection
- **Cost Control**: Detailed token usage and optimization recommendations
- **Quality Assurance**: Automated quality gates and human review workflows

### Proven Use Cases
- **Customer Support**: Monitor response quality and goal completion
- **Content Generation**: Detect hallucinations and maintain brand safety
- **Financial Services**: Compliance monitoring and bias detection
- **Healthcare**: Safety guardrails and factual accuracy verification

## Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLM Request   │───▶│  Quality Gates   │───▶│   Response      │
│                 │    │  - Hallucination │    │   Delivery      │
│                 │    │  - Toxicity      │    │                 │
│                 │    │  - Bias          │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Observability  │
                       │   - Traces       │
                       │   - Metrics      │
                       │   - Logs         │
                       │   - Feedback     │
                       └──────────────────┘
```

## Configuration

Create a `config.yaml` file:

```yaml
monitoring:
  quality_threshold: 0.8
  safety_checks:
    - hallucination
    - toxicity
    - bias
    - pii
  
cost_tracking:
  alert_threshold: 100.0  # USD per day
  
evaluation:
  batch_size: 100
  evaluation_metrics:
    - semantic_similarity
    - factual_accuracy
    - response_relevance
```

## Contributing

We welcome contributions that improve LLM quality and safety monitoring:

1. Quality evaluation methods
2. Safety detection algorithms  
3. Cost optimization techniques
4. Real-world use case examples

## License

MIT License - See LICENSE file for details

---

**Focus**: Quality, safety, and cost - the metrics that actually matter for production LLM applications.
