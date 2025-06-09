"""
LLM Quality & Safety Monitoring API

Focused API for monitoring LLM applications in production:
- Quality assessment endpoints
- Safety violation monitoring
- Cost tracking and optimization
- Real-time observability for LLM-specific metrics
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json

from monitoring.models import LLMTrace, QualityTrend, SafetyReport, CostAnalysis, AlertConfig
from monitoring.quality import QualityMonitor
from monitoring.cost import CostTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
quality_monitor = QualityMonitor()
cost_tracker = CostTracker()

# In-memory storage for metrics (would be a database in production)
quality_metrics = []
safety_assessments = []

# FastAPI app
app = FastAPI(
    title="LLM Quality & Safety Monitor",
    description="Production-ready API for monitoring LLM quality, safety, and costs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")

connection_manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting LLM Quality & Safety Monitor API")
    asyncio.create_task(broadcast_metrics_loop())

async def broadcast_metrics_loop():
    """Background task to broadcast quality metrics to WebSocket clients."""
    while True:
        try:
            if connection_manager.active_connections:
                # Broadcast quality trends
                cost_analysis = cost_tracker.get_cost_analysis("1h")
                await connection_manager.broadcast({
                    "type": "cost_update",
                    "data": cost_analysis.model_dump()
                })
            
            await asyncio.sleep(30)  # Broadcast every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(5)

# API Endpoints

@app.get("/health")
async def get_health():
    """Get service health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "LLM Quality & Safety Monitor",
        "version": "1.0.0"
    }

from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str
    response: str
    model_name: str = "unknown"
    user_id: Optional[str] = None
    check_hallucination: bool = True
    check_toxicity: bool = True
    check_bias: bool = True
    check_pii: bool = True

@app.post("/monitor/inference")
async def monitor_inference(request: InferenceRequest):
    """Monitor LLM inference request."""
    
    try:
        # Evaluate the response
        trace = quality_monitor.evaluate_response(
            prompt=request.prompt,
            response=request.response,
            model_name=request.model_name,
            check_hallucination=request.check_hallucination,
            check_toxicity=request.check_toxicity,
            check_bias=request.check_bias,
            check_pii=request.check_pii
        )
        
        # Store in metrics
        quality_metrics.append(trace.quality_metrics.model_dump())
        safety_assessments.append(trace.safety_assessment.model_dump())
        cost_tracker.log_inference(
            model=request.model_name,
            prompt_tokens=trace.cost_metrics.prompt_tokens,
            completion_tokens=trace.cost_metrics.completion_tokens
        )
        
        return {
            "trace_id": trace.trace_id,
            "quality_score": trace.quality_metrics.overall_quality,
            "safety_score": trace.safety_assessment.safety_score,
            "is_safe": trace.safety_assessment.is_safe,
            "safety_flags": [flag.value for flag in trace.safety_assessment.flags],
            "cost_usd": trace.cost_metrics.cost_usd,
            "timestamp": trace.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error processing inference monitoring: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics/quality")
async def get_quality_metrics(
    time_period: str = Query("24h", description="Time period for metrics")
):
    """Get quality metrics and trends."""
    try:
        # Mock quality trend data (would be calculated from stored traces)
        quality_trend = {
            "time_period": time_period,
            "average_quality": 0.85,
            "quality_decline": False,
            "decline_percentage": None,
            "top_issues": ["Response length too short", "Low factual accuracy"]
        }
        
        return quality_trend
        
    except Exception as e:
        logger.error(f"Error getting quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/safety")
async def get_safety_metrics(
    time_period: str = Query("24h", description="Time period for metrics")
):
    """Get safety violation metrics."""
    try:
        # Mock safety report data
        safety_report = {
            "time_period": time_period,
            "total_interactions": 1250,
            "safety_violations": 15,
            "violation_rate": 0.012,
            "common_flags": ["hallucination", "bias"],
            "critical_incidents": 2
        }
        
        return safety_report
        
    except Exception as e:
        logger.error(f"Error getting safety metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/cost")
async def get_cost_metrics(
    time_period: str = Query("24h", description="Time period for analysis")
):
    """Get cost analysis and optimization suggestions."""
    try:
        cost_analysis = cost_tracker.get_cost_analysis(time_period)
        return cost_analysis.model_dump()
        
    except Exception as e:
        logger.error(f"Error getting cost metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def batch_evaluate(
    evaluations: List[Dict[str, str]]
):
    """Batch evaluate multiple LLM responses."""
    try:
        results = []
        
        for eval_data in evaluations:
            prompt = eval_data.get("prompt", "")
            response = eval_data.get("response", "")
            model_name = eval_data.get("model_name", "unknown")
            
            if not prompt or not response:
                continue
                
            trace = quality_monitor.evaluate_response(
                prompt=prompt,
                response=response,
                model_name=model_name
            )
            
            results.append({
                "trace_id": trace.trace_id,
                "quality_score": trace.quality_metrics.overall_quality,
                "safety_score": trace.safety_assessment.safety_score,
                "is_safe": trace.safety_assessment.is_safe,
                "cost_usd": trace.cost_metrics.cost_usd
            })
        
        return {
            "evaluated_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback")
async def get_feedback_summary(
    time_period: str = Query("24h", description="Time period for feedback")
):
    """Get user feedback summary."""
    try:
        # Mock feedback data
        feedback_summary = {
            "time_period": time_period,
            "total_feedback": 350,
            "positive_feedback": 285,
            "negative_feedback": 65,
            "satisfaction_rate": 0.814,
            "common_complaints": [
                "Response too long",
                "Not relevant to question",
                "Factually incorrect"
            ],
            "improvement_suggestions": [
                "Shorten responses for simple questions",
                "Improve fact-checking mechanisms",
                "Better context understanding"
            ]
        }
        
        return feedback_summary
        
    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time quality and safety metrics."""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

def run_server():
    """Run the server."""
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    run_server() 