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
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
from pydantic import BaseModel

from monitoring.models import LLMTrace, QualityTrend, SafetyReport, CostAnalysis, AlertConfig, LLMTraceDB
from monitoring.quality import QualityMonitor
from monitoring.cost import CostTracker
from monitoring.database import engine, Base, SessionLocal

# Create database tables
Base.metadata.create_all(bind=engine)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
quality_monitor = QualityMonitor()
cost_tracker = CostTracker()

# FastAPI app
app = FastAPI(
    title="LLM Quality & Safety Monitor",
    description="Production-ready API for monitoring LLM quality, safety, and costs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
async def monitor_inference(request: InferenceRequest, db: SessionLocal = Depends(get_db)):
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
        
        # Store in DB
        db_trace = LLMTraceDB(
            trace_id=trace.trace_id,
            timestamp=trace.timestamp,
            prompt=trace.prompt,
            model_name=trace.model_name,
            user_id=trace.user_id,
            session_id=trace.session_id,
            response=trace.response,
            response_time_ms=trace.response_time_ms,
            semantic_similarity=trace.quality_metrics.semantic_similarity,
            factual_accuracy=trace.quality_metrics.factual_accuracy,
            response_relevance=trace.quality_metrics.response_relevance,
            coherence_score=trace.quality_metrics.coherence_score,
            overall_quality=trace.quality_metrics.overall_quality,
            is_safe=trace.safety_assessment.is_safe,
            safety_score=trace.safety_assessment.safety_score,
            safety_flags=[flag.value for flag in trace.safety_assessment.flags],
            prompt_tokens=trace.cost_metrics.prompt_tokens,
            completion_tokens=trace.cost_metrics.completion_tokens,
            total_tokens=trace.cost_metrics.total_tokens,
            cost_usd=trace.cost_metrics.cost_usd,
        )
        db.add(db_trace)
        db.commit()
        
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
    time_period: str = Query("24h", description="Time period for metrics"),
    db: SessionLocal = Depends(get_db)
):
    """Get quality metrics and trends."""
    try:
        end_date = datetime.now(timezone.utc)
        if time_period == "24h":
            start_date = end_date - timedelta(hours=24)
        elif time_period == "7d":
            start_date = end_date - timedelta(days=7)
        else:
            start_date = end_date - timedelta(days=30)
            
        traces = db.query(LLMTraceDB).filter(LLMTraceDB.timestamp >= start_date).all()
        
        if not traces:
            return {"message": "No data available for this time period."}
            
        avg_quality = sum(t.overall_quality for t in traces) / len(traces) if traces else 0
        
        return {
            "time_period": time_period,
            "average_quality": avg_quality,
            "total_evaluations": len(traces)
        }
        
    except Exception as e:
        logger.error(f"Error getting quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/safety")
async def get_safety_metrics(
    time_period: str = Query("24h", description="Time period for metrics"),
    db: SessionLocal = Depends(get_db)
):
    """Get safety violation metrics."""
    try:
        end_date = datetime.now(timezone.utc)
        if time_period == "24h":
            start_date = end_date - timedelta(hours=24)
        elif time_period == "7d":
            start_date = end_date - timedelta(days=7)
        else:
            start_date = end_date - timedelta(days=30)

        traces = db.query(LLMTraceDB).filter(LLMTraceDB.timestamp >= start_date).all()
        
        if not traces:
            return {"message": "No data available for this time period."}

        total_interactions = len(traces)
        safety_violations = sum(1 for t in traces if not t.is_safe)
        violation_rate = safety_violations / total_interactions if total_interactions > 0 else 0
        
        all_flags = []
        for t in traces:
            if t.safety_flags:
                all_flags.extend(t.safety_flags)
        
        from collections import Counter
        common_flags = [item[0] for item in Counter(all_flags).most_common(3)]

        return {
            "time_period": time_period,
            "total_interactions": total_interactions,
            "safety_violations": safety_violations,
            "violation_rate": violation_rate,
            "common_flags": common_flags
        }
        
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
    evaluations: List[Dict[str, str]],
    db: SessionLocal = Depends(get_db)
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
            
            # Store in DB
            db_trace = LLMTraceDB(
                trace_id=trace.trace_id,
                timestamp=trace.timestamp,
                prompt=trace.prompt,
                model_name=trace.model_name,
                response=trace.response,
                response_time_ms=trace.response_time_ms,
                semantic_similarity=trace.quality_metrics.semantic_similarity,
                factual_accuracy=trace.quality_metrics.factual_accuracy,
                response_relevance=trace.quality_metrics.response_relevance,
                coherence_score=trace.quality_metrics.coherence_score,
                overall_quality=trace.quality_metrics.overall_quality,
                is_safe=trace.safety_assessment.is_safe,
                safety_score=trace.safety_assessment.safety_score,
                safety_flags=[flag.value for flag in trace.safety_assessment.flags],
                prompt_tokens=trace.cost_metrics.prompt_tokens,
                completion_tokens=trace.cost_metrics.completion_tokens,
                total_tokens=trace.cost_metrics.total_tokens,
                cost_usd=trace.cost_metrics.cost_usd,
            )
            db.add(db_trace)

            results.append({
                "trace_id": trace.trace_id,
                "quality_score": trace.quality_metrics.overall_quality,
                "safety_score": trace.safety_assessment.safety_score,
                "is_safe": trace.safety_assessment.is_safe,
                "cost_usd": trace.cost_metrics.cost_usd
            })
        
        db.commit()
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