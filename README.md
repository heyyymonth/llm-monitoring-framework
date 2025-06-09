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

## Real-World Use Cases

### Financial Services
- Bias detection for loan decisions
- Compliance monitoring for regulatory requirements
- Audit trails for model decisions
- PII protection and data governance

### Healthcare
- Safety guardrails for medical advice
- Factual accuracy verification
- HIPAA compliance monitoring
- Patient safety incident tracking

### Customer Service
- Response quality and goal completion
- User satisfaction correlation
- Cost per resolution tracking
- Escalation pattern analysis

### Enterprise
- Prompt injection attack detection
- Data leakage prevention
- Cost optimization across departments
- Quality gates for brand protection

## Business Impact & ROI

### Before vs After
- ✅ **Focus on metrics that actually matter** (not CPU/memory)
- ✅ **Production-ready quality assurance** with automated gates
- ✅ **Cost optimization and budget control** preventing overruns
- ✅ **Enterprise compliance capabilities** for regulated industries
- ✅ **Real-world use case alignment** with proven business value

### ROI Potential
- **Cost Savings**: 20-40% reduction through optimization
- **Quality Improvement**: Early detection prevents reputation damage
- **Compliance**: Avoid regulatory fines and audit failures
- **Productivity**: Automated quality gates reduce manual review

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

## Core Components

1. **Quality Monitoring Module** (`monitoring/quality.py`)
   - Hallucination detection algorithms
   - Safety evaluation (toxicity, bias, PII)
   - Response quality assessment
   - Coherence and relevance scoring

2. **Cost Tracking Module** (`monitoring/cost.py`)
   - Token usage monitoring
   - Cost analysis and optimization
   - Budget alerts and projections
   - Model efficiency comparison

3. **Modern Data Models** (`monitoring/models.py`)
   - LLM-specific trace structures
   - Quality and safety metrics
   - Cost analysis models
   - Alert configuration schemas

4. **Production API** (`api/server.py`)
   - Quality assessment endpoints
   - Safety violation monitoring
   - Cost optimization insights
   - Real-time observability

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

## Roadmap

### Phase 1: Core Enhancement (Next 30 days)
- [ ] Advanced hallucination detection using embeddings
- [ ] Integration with popular LLM providers (OpenAI, Anthropic, etc.)
- [ ] Comprehensive dashboard with quality visualizations
- [ ] Prompt optimization recommendations

### Phase 2: Enterprise Features (Next 60 days)
- [ ] RBAC and multi-tenant support
- [ ] Compliance reporting for SOX, GDPR, HIPAA
- [ ] Audit trail and data retention policies
- [ ] A/B testing framework for prompt optimization

### Phase 3: Advanced Analytics (Next 90 days)
- [ ] ML-based drift detection and alerting
- [ ] Predictive cost modeling and budgeting
- [ ] Advanced bias detection using fairness metrics
- [ ] Integration with popular MLOps platforms

### Phase 4: AI-Powered Optimization (Next 120 days)
- [ ] Automated prompt optimization using RL
- [ ] Intelligent model routing for cost efficiency
- [ ] Predictive quality scoring
- [ ] Anomaly detection for safety violations

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
