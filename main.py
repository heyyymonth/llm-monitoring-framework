#!/usr/bin/env python3
"""
LLM Quality & Safety Monitor

Production-ready monitoring framework for LLM applications focusing on:
- Quality assessment and drift detection
- Safety guardrails and violation monitoring  
- Cost tracking and optimization
- Real-time observability for LLM-specific metrics

Usage:
    python main.py                    # Start full monitoring stack
    python main.py --api-only         # Start API server only
    python main.py --dashboard-only   # Start dashboard only
"""

import subprocess
import sys
import time
import signal
import os
import argparse
from pathlib import Path

def start_api_server():
    """Start the LLM monitoring API server on port 8000."""
    print("🚀 Starting LLM Quality & Safety Monitoring API on http://localhost:8000")
    cmd = [sys.executable, "-m", "uvicorn", "api.server:app", 
           "--host", "0.0.0.0", "--port", "8000", "--reload"]
    return subprocess.Popen(cmd)

def start_dashboard():
    """Start the LLM monitoring dashboard on port 8080."""
    print("📊 Starting LLM Quality & Safety Dashboard on http://localhost:8080")
    cmd = [sys.executable, "dashboard/app.py"]
    return subprocess.Popen(cmd)

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['fastapi', 'uvicorn', 'dash', 'plotly', 'pandas', 'pydantic']
    missing = []
    
    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main entry point for LLM Quality & Safety Monitor."""
    
    parser = argparse.ArgumentParser(description="LLM Quality & Safety Monitor")
    parser.add_argument("--api-only", action="store_true", 
                       help="Start API server only")
    parser.add_argument("--dashboard-only", action="store_true", 
                       help="Start dashboard only")
    
    args = parser.parse_args()
    
    print("🧠 LLM Quality & Safety Monitor")
    print("=" * 60)
    print("Focus: Quality, Safety, and Cost - the metrics that matter")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Verify required files exist
    required_files = [
        "api/server.py",
        "monitoring/models.py",
        "monitoring/quality.py",
        "monitoring/cost.py"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)
    
    processes = []
    
    try:
        if args.api_only:
            print("\n🔧 Running in API-only mode")
            api_process = start_api_server()
            processes.append(api_process)
        elif args.dashboard_only:
            print("\n🔧 Running in dashboard-only mode")
            dashboard_process = start_dashboard()
            processes.append(dashboard_process)
        else:
            # Start both services
            api_process = start_api_server()
            processes.append(api_process)
            time.sleep(2)  # Give API time to start
            
            dashboard_process = start_dashboard()
            processes.append(dashboard_process)
            time.sleep(2)  # Give dashboard time to start
        
        print("\n✅ LLM Quality & Safety Monitor started successfully!")
        print("━" * 60)
        print("📖 API Documentation: http://localhost:8000/docs")
        print("📊 Dashboard: http://localhost:8080")
        print("🔧 Health Check: http://localhost:8000/health")
        print("🎯 Monitor Inference: POST http://localhost:8000/monitor/inference")
        print("📈 Quality Metrics: GET http://localhost:8000/metrics/quality")
        print("🛡️  Safety Metrics: GET http://localhost:8000/metrics/safety")
        print("💰 Cost Analysis: GET http://localhost:8000/metrics/cost")
        print("━" * 60)
        print("💡 Press Ctrl+C to stop all services")
        
        # Wait for processes
        try:
            while True:
                # Check if processes are still running
                for i, proc in enumerate(processes):
                    if proc.poll() is not None:
                        print(f"⚠️  Process {i} stopped unexpectedly")
                        return
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down LLM Quality & Safety Monitor...")
    
    except Exception as e:
        print(f"❌ Error starting services: {e}")
    
    finally:
        # Clean shutdown
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 