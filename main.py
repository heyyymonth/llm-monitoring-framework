#!/usr/bin/env python3
"""
LLM Performance Monitoring Framework
Main entry point for starting the monitoring system.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
import subprocess
import threading
import time

from monitoring.config import get_config, load_config
from api.server import run_server
from dashboard.app import run_dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal, stopping services...")
    sys.exit(0)


def start_api_server():
    """Start the API server in a separate process."""
    try:
        run_server()
    except Exception as e:
        logger.error(f"API server error: {e}")


def start_dashboard():
    """Start the dashboard in a separate process."""
    try:
        run_dashboard()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")


def run_api_only():
    """Run only the API server."""
    logger.info("Starting LLM Monitoring API server...")
    start_api_server()


def run_dashboard_only():
    """Run only the dashboard."""
    logger.info("Starting LLM Monitoring Dashboard...")
    start_dashboard()


def run_all():
    """Run both API server and dashboard."""
    config = get_config()
    
    logger.info("Starting LLM Performance Monitoring Framework")
    logger.info(f"API will be available at: http://{config.api.host}:{config.api.port}")
    logger.info(f"Dashboard will be available at: http://{config.dashboard.host}:{config.dashboard.port}")
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Start dashboard in main thread
    start_dashboard()


def run_example():
    """Run example usage of the monitoring framework."""
    import uuid
    from monitoring.client import LLMMonitor
    
    logger.info("Running monitoring framework example...")
    
    # Create monitor client
    monitor = LLMMonitor("http://localhost:8000")
    
    # Simulate some LLM inference requests
    for i in range(10):
        request_id = str(uuid.uuid4())
        
        with monitor.track_request(
            model_name="example-model", 
            request_id=request_id
        ) as tracker:
            # Simulate processing
            import time
            import random
            
            # Set prompt info
            prompt_tokens = random.randint(10, 100)
            prompt_length = prompt_tokens * 4  # Rough estimate
            
            tracker.set_prompt_info(
                tokens=prompt_tokens,
                length=prompt_length,
                temperature=0.7,
                max_tokens=150
            )
            
            # Simulate queue time
            time.sleep(random.uniform(0.01, 0.1))
            tracker.start_processing()
            
            # Simulate processing time
            processing_time = random.uniform(0.5, 2.0)
            time.sleep(processing_time)
            
            # Simulate occasional errors
            if random.random() < 0.1:
                raise Exception("Simulated inference error")
            
            # Set response info
            completion_tokens = random.randint(20, 150)
            response_length = completion_tokens * 4
            
            tracker.set_response_info(
                tokens=completion_tokens,
                length=response_length
            )
            
            # Add metadata
            tracker.set_metadata(
                finish_reason="stop",
                model_version="1.0.0"
            )
        
        logger.info(f"Completed request {i+1}/10: {request_id}")
        time.sleep(random.uniform(0.1, 0.5))
    
    logger.info("Example completed! Check the dashboard at http://localhost:8080")


def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'psutil', 'redis', 'pandas', 'numpy',
        'plotly', 'dash', 'sqlalchemy', 'pydantic', 'httpx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run with --install to install missing dependencies")
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Performance Monitoring Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start both API and dashboard
  python main.py --api-only         # Start only the API server
  python main.py --dashboard-only   # Start only the dashboard
  python main.py --example          # Run monitoring example
  python main.py --install          # Install dependencies
        """
    )
    
    parser.add_argument(
        '--api-only',
        action='store_true',
        help='Start only the API server'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Start only the dashboard'
    )
    
    parser.add_argument(
        '--example',
        action='store_true',
        help='Run example usage of the monitoring framework'
    )
    
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install required dependencies'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Handle install command
    if args.install:
        install_dependencies()
        return
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Run with --install to install them.")
        sys.exit(1)
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        from monitoring.config import set_config
        set_config(config)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.example:
            run_example()
        elif args.api_only:
            run_api_only()
        elif args.dashboard_only:
            run_dashboard_only()
        else:
            run_all()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 