# Contributing to LLM Performance Monitoring Framework

Thank you for your interest in contributing to the LLM Performance Monitoring Framework! We welcome contributions from developers of all skill levels.

## 📋 **Table of Contents**

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Areas for Contribution](#areas-for-contribution)
- [Community Guidelines](#community-guidelines)

---

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- Git
- Basic understanding of FastAPI, Pydantic, and async Python
- Familiarity with LLM concepts (helpful but not required)

### **First Steps**
1. **Fork the repository** on GitHub
2. **Star the repository** to show your support
3. **Read the documentation** in the `docs/` folder
4. **Set up your development environment** (see below)
5. **Look for "good first issue" labels** in the GitHub Issues

---

## 💻 **Development Setup**

### **1. Clone Your Fork**
```bash
git clone https://github.com/your-username/llm-monitoring-framework.git
cd llm-monitoring-framework
git remote add upstream https://github.com/original-owner/llm-monitoring-framework.git
```

### **2. Create Development Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install -e .  # Editable install
```

### **3. Verify Setup**
```bash
# Run basic tests
python examples/integrations/quick_test.py

# Run full test suite
python tests/run_tests.py

# Start services (in separate terminals)
python main.py
```

### **4. Development Tools**
```bash
# Code formatting
pip install black isort flake8

# Type checking
pip install mypy

# Testing tools
pip install pytest pytest-cov
```

---

## 📝 **Contributing Guidelines**

### **Types of Contributions**

#### **🐛 Bug Reports**
- Use the GitHub Issues template
- Include detailed reproduction steps
- Provide system information and error logs
- Test with the latest version first

#### **✨ Feature Requests**
- Check existing issues for duplicates
- Describe the use case and benefits
- Consider backward compatibility
- Provide implementation ideas if possible

#### **📖 Documentation**
- Fix typos and improve clarity
- Add examples and use cases
- Update API documentation
- Translate to other languages

#### **🔧 Code Contributions**
- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

### **Before You Start**
1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for significant changes
3. **Discuss your approach** with maintainers
4. **Keep changes focused** - one feature/fix per PR

---

## 🎯 **Code Standards**

### **Python Style Guide**
We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Line length: 88 characters (Black default)
# Use type hints for all functions
def process_metrics(data: List[InferenceMetrics]) -> PerformanceSummary:
    """Process metrics with proper type annotations."""
    pass

# Use docstrings for all public functions
def track_inference(self, model_name: str) -> InferenceTracker:
    """
    Track an LLM inference request.
    
    Args:
        model_name: Name of the LLM model being used
        
    Returns:
        InferenceTracker: Context manager for tracking the request
    """
    pass
```

### **Code Formatting**
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check with flake8
flake8 .

# Type checking
mypy monitoring/ api/ dashboard/
```

### **Naming Conventions**
- **Classes**: `PascalCase` (e.g., `InferenceMetrics`)
- **Functions/Variables**: `snake_case` (e.g., `track_inference`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private methods**: `_snake_case` (e.g., `_collect_metrics`)

### **Project Structure**
```
monitoring/           # Core monitoring package
├── models.py        # Pydantic data models
├── metrics.py       # Metrics collection
├── database.py      # Data storage
├── alerts.py        # Alert management
├── client.py        # Client SDK
└── config.py        # Configuration management

api/                 # FastAPI server
└── server.py        # API endpoints

dashboard/           # Web dashboard
└── app.py          # Dash application

tests/               # Test suite
├── unit/           # Unit tests
├── integration/    # Integration tests
└── run_tests.py    # Test runner

examples/            # Example integrations
└── integrations/   # LLM integration examples

docs/               # Documentation
```

---

## 🧪 **Testing Requirements**

### **Test Categories**

#### **Unit Tests** (`tests/unit/`)
- Test individual functions and classes
- Mock external dependencies
- Fast execution (< 5 seconds total)
- High coverage (>80% for new code)

```python
def test_inference_metrics_creation():
    """Test creating inference metrics."""
    metrics = InferenceMetrics(
        request_id="test-123",
        model_name="test-model",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        response_time_ms=1500.0
    )
    
    assert metrics.request_id == "test-123"
    assert metrics.success is True
```

#### **Integration Tests** (`tests/integration/`)
- Test component interactions
- Real database operations
- End-to-end workflows
- Performance validation

#### **LLM Integration Tests** (`examples/integrations/`)
- Real LLM testing (optional)
- Framework compatibility
- Performance benchmarks
- Error handling

### **Running Tests**
```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Run with coverage
coverage run -m pytest tests/
coverage html
coverage report

# Run performance tests
python examples/integrations/test_ollama_llm.py
```

### **Test Requirements for PRs**
- ✅ All existing tests must pass
- ✅ New features require tests
- ✅ Bug fixes require regression tests
- ✅ Coverage should not decrease significantly
- ✅ Tests should be fast and reliable

---

## 📤 **Submitting Changes**

### **Pull Request Process**

#### **1. Prepare Your Changes**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code, test, document ...

# Format and test
black .
isort .
python tests/run_tests.py

# Commit changes
git add .
git commit -m "Add your feature description"
```

#### **2. Submit Pull Request**
```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
# Fill out the PR template
# Link to relevant issues
```

#### **3. PR Review Process**
- **Automated checks** will run (tests, linting)
- **Maintainer review** within 2-3 business days
- **Address feedback** promptly
- **Squash commits** if requested
- **Merge** after approval

### **Commit Message Guidelines**
```bash
# Format: <type>(<scope>): <description>

feat(monitoring): add GPU temperature monitoring
fix(client): handle connection timeout gracefully
docs(readme): update installation instructions
test(alerts): add tests for custom alert rules
refactor(api): improve error handling structure
```

### **PR Title Format**
- `feat: Add GPU temperature monitoring`
- `fix: Handle Redis connection failures`
- `docs: Update API documentation`
- `test: Add integration tests for Ollama`

---

## 🎯 **Areas for Contribution**

### **🚨 High Priority**
- **Docker support** - Containerization and docker-compose
- **Authentication** - User management and access control
- **Webhook alerts** - External system notifications
- **Performance optimization** - Caching and batch processing

### **📊 Medium Priority**
- **Additional LLM integrations** (Anthropic, Cohere, etc.)
- **Enhanced visualizations** - More chart types and customization
- **Export capabilities** - PDF reports, advanced data export
- **Cost tracking** - Token pricing and budget monitoring

### **🔧 Good First Issues**
- **Documentation improvements** - Fix typos, add examples
- **Test coverage** - Add tests for existing functionality
- **Error messages** - Improve error handling and messages
- **Configuration options** - Add new configuration parameters

### **🌟 Advanced Contributions**
- **Multi-tenant support** - Organization and team management
- **Content analysis** - Toxicity detection, sentiment analysis
- **A/B testing** - Model comparison and experimentation
- **Real-time streaming** - Enhanced WebSocket functionality

---

## 👥 **Community Guidelines**

### **Code of Conduct**
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other community members

### **Communication Channels**
- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - General questions, ideas
- **Pull Requests** - Code review and collaboration

### **Getting Help**
- **Documentation** - Check `docs/` folder first
- **Examples** - Look at `examples/integrations/`
- **Issues** - Search existing issues
- **Create Issue** - If you can't find an answer

### **Recognition**
Contributors will be recognized in:
- **README.md** - Contributors section
- **CHANGELOG.md** - Feature attribution
- **Release notes** - Major contribution highlights

---

## 🏆 **Contributor Levels**

### **🌱 First-time Contributors**
- Documentation fixes
- Simple bug fixes
- Test additions
- Code formatting improvements

### **🚀 Regular Contributors**
- Feature implementations
- Significant bug fixes
- Architecture improvements
- Performance optimizations

### **⭐ Core Contributors**
- Design decisions
- Code reviews
- Release management
- Community leadership

---

## 📞 **Questions?**

- **Documentation**: Check the `docs/` folder
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: [maintainer-email@domain.com]

**Thank you for contributing to the LLM Performance Monitoring Framework! 🎉** 