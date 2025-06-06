import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json

from monitoring.models import InferenceMetrics, SystemMetrics, PerformanceSummary, HealthStatus
from monitoring.metrics import MetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
metrics_collector = MetricsCollector()

# FastAPI app
app = FastAPI(
    title="LLM Performance Monitor",
    description="Minimalist LLM performance monitoring API",
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
    logger.info("Starting LLM Performance Monitor API")
    metrics_collector.start()
    asyncio.create_task(broadcast_metrics_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LLM Performance Monitor API")
    metrics_collector.stop()

async def broadcast_metrics_loop():
    """Background task to broadcast metrics to WebSocket clients."""
    while True:
        try:
            if connection_manager.active_connections:
                current_metrics = metrics_collector.get_current_system_metrics()
                if current_metrics:
                    await connection_manager.broadcast({
                        "type": "system_metrics",
                        "data": current_metrics.model_dump()
                    })
                
                recent_inferences = metrics_collector.get_recent_inference_metrics(10)
                if recent_inferences:
                    await connection_manager.broadcast({
                        "type": "inference_metrics",
                        "data": [m.model_dump() for m in recent_inferences]
                    })
            
            await asyncio.sleep(2)  # Broadcast every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(5)

# API Endpoints

@app.get("/health")
async def get_health():
    """Get service health status."""
    system_metrics = metrics_collector.get_current_system_metrics()
    performance_summary = metrics_collector.get_performance_summary("1h")
    
    status = "healthy"
    if system_metrics:
        if (system_metrics.cpu_percent > 90 or 
            system_metrics.memory_percent > 95 or
            performance_summary.error_rate > 10):
            status = "degraded"
        if (system_metrics.cpu_percent > 98 or 
            system_metrics.memory_percent > 98 or
            performance_summary.error_rate > 25):
            status = "unhealthy"
    
    health_status = HealthStatus(
        status=status,
        uptime_seconds=metrics_collector.get_uptime(),
        error_rate_1h=performance_summary.error_rate,
        avg_response_time_1h=performance_summary.avg_response_time_ms
    )
    
    return health_status.model_dump()

@app.post("/track/inference")
async def track_inference(metrics: InferenceMetrics):
    """Track inference metrics."""
    try:
        metrics_collector.log_inference(metrics)
        return {"status": "success", "request_id": metrics.request_id}
    except Exception as e:
        logger.error(f"Error tracking inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/current")
async def get_current_metrics():
    """Get current system metrics."""
    try:
        system_metrics = metrics_collector.get_current_system_metrics()
        performance_summary = metrics_collector.get_performance_summary("1h")
        
        return {
            "system": system_metrics.model_dump() if system_metrics else None,
            "performance": performance_summary.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/history")
async def get_metrics_history(
    metric_type: str = Query("inference", description="Type of metrics (inference/system)"),
    hours: int = Query(1, description="Hours of history"),
    limit: int = Query(100, description="Max results")
):
    """Get metrics history."""
    try:
        if metric_type == "inference":
            metrics = metrics_collector.get_recent_inference_metrics(limit)
            return [m.model_dump() for m in metrics]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/summary")
async def get_performance_summary(time_period: str = Query("1h", description="Time period")):
    """Get performance summary."""
    try:
        summary = metrics_collector.get_performance_summary(time_period)
        return summary.model_dump()
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get basic stats."""
    try:
        return metrics_collector.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
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
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 