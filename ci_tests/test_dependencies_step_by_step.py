#!/usr/bin/env python3
"""
Step-by-step dependency test to isolate CI issues.
Tests each dependency one by one to find the exact failure point.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_dependency(dep_name, import_statement):
    """Test a single dependency."""
    try:
        exec(import_statement)
        print(f"✅ {dep_name}")
        return True
    except Exception as e:
        print(f"❌ {dep_name}: {e}")
        return False

def main():
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print("\n🔍 Testing dependencies step by step:\n")
    
    tests = [
        ("Standard typing", "from typing import Dict, Any, List, Optional"),
        ("Standard datetime", "from datetime import datetime, timezone"),
        ("Standard asyncio", "import asyncio"),
        ("Standard logging", "import logging"),
        ("Pydantic BaseModel", "from pydantic import BaseModel"),
        ("Pydantic Field", "from pydantic import Field"),
        ("FastAPI core", "from fastapi import FastAPI"),
        ("FastAPI HTTPException", "from fastapi import HTTPException"),
        ("FastAPI Query", "from fastapi import Query"),
        ("CORS middleware", "from fastapi.middleware.cors import CORSMiddleware"),
        ("JSON Response", "from fastapi.responses import JSONResponse"),
        ("Uvicorn", "import uvicorn"),
        ("Project monitoring module", "import monitoring"),
        ("Project api module", "import api"),
        ("LLM Trace model", "from monitoring.models import LLMTrace"),
        ("Quality models", "from monitoring.models import QualityTrend, SafetyReport"),
        ("Cost models", "from monitoring.models import CostAnalysis, AlertConfig"),
        ("Quality Monitor", "from monitoring.quality import QualityMonitor"),
        ("Cost Tracker", "from monitoring.cost import CostTracker"),
    ]
    
    failed = []
    for name, stmt in tests:
        if not test_dependency(name, stmt):
            failed.append(name)
    
    print(f"\n📊 Results: {len(tests) - len(failed)}/{len(tests)} passed")
    
    if failed:
        print(f"❌ Failed dependencies: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("🎉 All dependencies working!")

if __name__ == '__main__':
    main() 