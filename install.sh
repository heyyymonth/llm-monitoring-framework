#!/bin/bash

# LLM Performance Monitoring Framework - Installation Script
# ==========================================================

set -e  # Exit on any error

echo "🚀 Installing LLM Performance Monitoring Framework..."

# Check Python version
if ! python3 --version &> /dev/null; then
    echo "❌ Python 3.8+ is required but not found"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs screenshots

# Set up configuration
if [ ! -f "config.yaml" ]; then
    echo "⚙️  Creating default configuration..."
    cp config.yaml config.local.yaml
fi

# Test installation
echo "🧪 Testing installation..."
python -c "import monitoring; print('✅ Core monitoring package imported successfully')"

# Run quick functionality test
if [ -f "examples/integrations/quick_test.py" ]; then
    echo "🔬 Running functionality test..."
    python examples/integrations/quick_test.py
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📖 Next steps:"
echo "   1. Start services: python main.py"
echo "   2. Open dashboard: http://localhost:8080"
echo "   3. Check API docs: http://localhost:8000/docs"
echo "   4. Run tests: python tests/run_tests.py"
echo "" 