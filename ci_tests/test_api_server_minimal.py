#!/usr/bin/env python3
"""
Minimal API server validation script for CI testing.
Isolates import issues by testing components separately.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python and standard library imports."""
    try:
        import asyncio
        import logging
        from datetime import datetime, timezone
        from typing import Dict, Any, List, Optional
        print("✅ Standard library imports successful")
        return True
    except Exception as e:
        print(f"❌ Standard library import failed: {e}")
        return False

def test_fastapi_imports():
    """Test FastAPI related imports."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        import uvicorn
        print("✅ FastAPI imports successful")
        return True
    except Exception as e:
        print(f"❌ FastAPI import failed: {e}")
        traceback.print_exc()
        return False

def test_monitoring_imports():
    """Test monitoring module imports."""
    try:
        from monitoring.models import LLMTrace, QualityTrend, SafetyReport, CostAnalysis, AlertConfig
        print("✅ monitoring.models imported")
        
        from monitoring.quality import QualityMonitor
        print("✅ monitoring.quality imported")
        
        from monitoring.cost import CostTracker
        print("✅ monitoring.cost imported")
        return True
    except Exception as e:
        print(f"❌ Monitoring imports failed: {e}")
        traceback.print_exc()
        return False

def test_simple_fastapi_app():
    """Test creating a simple FastAPI app without monitoring dependencies."""
    try:
        from fastapi import FastAPI
        app = FastAPI(title="Test App")
        print("✅ Simple FastAPI app created")
        return True
    except Exception as e:
        print(f"❌ Simple FastAPI app creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    print(f"🐍 Python version: {sys.version}")
    print(f"🔧 Platform: {sys.platform}")
    
    success = True
    
    # Test each component separately
    success &= test_basic_imports()
    success &= test_fastapi_imports()
    success &= test_monitoring_imports()
    success &= test_simple_fastapi_app()
    
    if success:
        print("🎉 All components work individually - testing full API server...")
        try:
            # Only import the full API server if all components work
            from api.server import app
            print('✅ Full API server import successful')
        except Exception as e:
            print(f'❌ Full API server import failed: {e}')
            traceback.print_exc()
            sys.exit(1)
    else:
        print("❌ Component tests failed - skipping full API server test")
        sys.exit(1)

if __name__ == '__main__':
    main() 