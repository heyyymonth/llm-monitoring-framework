#!/usr/bin/env python3
"""
Minimalist LLM Performance Monitor
Simple launcher for API server and dashboard.
"""

import subprocess
import sys
import time
import signal
import os
import threading
from pathlib import Path

def start_api_server():
    """Start the API server on port 8000."""
    print("🚀 Starting API server on http://localhost:8000")
    cmd = [sys.executable, "-m", "uvicorn", "api.server:app", 
           "--host", "0.0.0.0", "--port", "8000", "--reload"]
    return subprocess.Popen(cmd)

def start_dashboard():
    """Start the dashboard on port 8080."""
    print("📊 Starting dashboard on http://localhost:8080")
    cmd = [sys.executable, "dashboard/app.py"]
    return subprocess.Popen(cmd)

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['fastapi', 'uvicorn', 'dash', 'psutil', 'plotly', 'pandas']
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
    """Main entry point."""
    print("🧠 Minimalist LLM Performance Monitor")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Verify required files exist
    required_files = [
        "api/server.py",
        "dashboard/app.py",
        "monitoring/metrics.py",
        "monitoring/models.py"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)
    
    processes = []
    
    try:
        # Start API server
        api_process = start_api_server()
        processes.append(api_process)
        time.sleep(2)  # Give API time to start
        
        # Start dashboard
        dashboard_process = start_dashboard()
        processes.append(dashboard_process)
        time.sleep(2)  # Give dashboard time to start
        
        print("\n✅ Services started successfully!")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("📊 Dashboard: http://localhost:8080")
        print("🔧 Health Check: http://localhost:8000/health")
        print("\n💡 Press Ctrl+C to stop all services")
        
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
            print("\n🛑 Shutting down services...")
    
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