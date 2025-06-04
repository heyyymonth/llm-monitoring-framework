#!/bin/bash
# Installation script for LLM Performance Monitoring Framework

set -e

echo "🚀 Installing LLM Performance Monitoring Framework"
echo "=================================================="

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc) -eq 0 ]]; then
    echo "❌ Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p logs

# Check if Redis is available (optional)
echo "🔍 Checking Redis availability..."
if command -v redis-server &> /dev/null; then
    echo "✅ Redis server found"
    if ! pgrep -x "redis-server" > /dev/null; then
        echo "⚠️  Redis server not running. You may want to start it:"
        echo "   redis-server"
    else
        echo "✅ Redis server is running"
    fi
else
    echo "⚠️  Redis server not found. Install with:"
    echo "   # Ubuntu/Debian:"
    echo "   sudo apt-get install redis-server"
    echo "   # macOS:"
    echo "   brew install redis"
    echo "   # Or use Docker:"
    echo "   docker run -d -p 6379:6379 redis:alpine"
fi

# Run basic health check
echo "🔍 Running health check..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    from monitoring.config import get_config
    from monitoring.models import SystemMetrics
    from monitoring.metrics import MetricsCollector
    print('✅ All core modules imported successfully')
except Exception as e:
    print(f'❌ Error importing modules: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📖 Quick Start:"
echo "   # Start the monitoring system:"
echo "   python main.py"
echo ""
echo "   # Or start components separately:"
echo "   python main.py --api-only       # API server only"
echo "   python main.py --dashboard-only # Dashboard only"
echo ""
echo "   # Run example:"
echo "   python main.py --example"
echo ""
echo "   # Install dependencies if missing:"
echo "   python main.py --install"
echo ""
echo "🌐 URLs:"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Dashboard:         http://localhost:8080"
echo ""
echo "📚 For more information, see README.md" 