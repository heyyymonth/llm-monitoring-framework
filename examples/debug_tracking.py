#!/usr/bin/env python3
"""
Debug tracking script to test the monitoring framework.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.client import create_monitor

def debug_tracking():
    """Simple tracking test with debug output."""
    
    print("🔍 Creating monitor...")
    monitor = create_monitor("http://localhost:8000")
    
    print("🔍 Starting tracking context...")
    with monitor.track_request(model_name="debug-test") as tracker:
        print("🔍 Inside tracking context")
        
        # Set prompt info
        tracker.set_prompt_info(tokens=10, length=50)
        print("🔍 Set prompt info")
        
        # Start processing
        tracker.start_processing()
        print("🔍 Started processing")
        
        # Simulate some work
        time.sleep(1)
        
        # Set response info
        tracker.set_response_info(tokens=20, length=100)
        print("🔍 Set response info")
        
        print("🔍 About to exit context (this should trigger log_completion)")
    
    print("🔍 Exited tracking context")
    
    # Give it a moment to send metrics
    print("🔍 Waiting for metrics to be sent...")
    time.sleep(2)
    
    print("🔍 Debug tracking complete")

if __name__ == "__main__":
    debug_tracking() 