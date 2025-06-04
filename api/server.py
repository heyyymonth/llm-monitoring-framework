import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json

from monitoring.config import get_config
from monitoring.models import (
    InferenceMetrics, ErrorMetrics, SystemMetrics, PerformanceSummary,
    HealthStatus, ModelInfo, QueueMetrics
)
from monitoring.metrics import MetricsCollector
from monitoring.database import DatabaseManager
from monitoring.alerts import AlertManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
config = get_config()
metrics_collector = MetricsCollector()
database_manager = DatabaseManager()
alert_manager = AlertManager(metrics_collector, database_manager)

# FastAPI app
app = FastAPI(
    title="LLM Performance Monitor API",
    description="API for monitoring LLM performance and actions",
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
active_connections: List[WebSocket] = []


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")


connection_manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting LLM Performance Monitor API")
    
    # Start metrics collection
    metrics_collector.start()
    
    # Start alert manager
    alert_manager.start()
    
    # Start background tasks
    asyncio.create_task(broadcast_metrics_loop())
    asyncio.create_task(cleanup_old_data_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LLM Performance Monitor API")
    
    # Stop services
    metrics_collector.stop()
    alert_manager.stop()


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
                
                # Broadcast recent inference metrics
                recent_inferences = metrics_collector.get_recent_inference_metrics(10)
                if recent_inferences:
                    await connection_manager.broadcast({
                        "type": "inference_metrics",
                        "data": [m.model_dump() for m in recent_inferences]
                    })
            
            await asyncio.sleep(1)  # Broadcast every second
            
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(5)


async def cleanup_old_data_loop():
    """Background task to cleanup old data."""
    while True:
        try:
            # Cleanup every hour
            await asyncio.sleep(3600)
            
            # Cleanup old metrics from memory
            metrics_collector.cleanup_old_metrics()
            
            # Cleanup old data from database
            database_manager.cleanup_old_data(config.monitoring.max_history_days)
            
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")


# API Endpoints

@app.get("/health")
async def get_health():
    """Get service health status."""
    system_metrics = metrics_collector.get_current_system_metrics()
    performance_summary = metrics_collector.get_performance_summary("1h")
    db_stats = database_manager.get_database_stats()
    
    # Determine health status
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
        version="1.0.0",
        uptime_seconds=metrics_collector.get_uptime(),
        error_rate_1h=performance_summary.error_rate,
        avg_response_time_1h=performance_summary.avg_response_time_ms,
        system_metrics=system_metrics or SystemMetrics(
            cpu_percent=0, memory_percent=0, memory_used_gb=0,
            memory_total_gb=0, disk_percent=0
        ),
        services={
            "database": db_stats.get("redis_connected", False),
            "metrics_collector": metrics_collector._running,
            "alert_manager": alert_manager._running
        }
    )
    
    return health_status.model_dump()


@app.post("/track/inference")
async def track_inference(metrics: InferenceMetrics, background_tasks: BackgroundTasks):
    """Track inference metrics."""
    try:
        # Log to metrics collector
        metrics_collector.log_inference(metrics)
        
        # Store in database (background task)
        background_tasks.add_task(database_manager.store_inference_metrics, metrics)
        
        return {"status": "success", "request_id": metrics.request_id}
    
    except Exception as e:
        logger.error(f"Error tracking inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track/error")
async def track_error(metrics: ErrorMetrics, background_tasks: BackgroundTasks):
    """Track error metrics."""
    try:
        # Log to metrics collector
        metrics_collector.log_error(metrics)
        
        # Store in database (background task)
        background_tasks.add_task(database_manager.store_error_metrics, metrics)
        
        return {"status": "success", "request_id": metrics.request_id}
    
    except Exception as e:
        logger.error(f"Error tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track/custom")
async def track_custom_metric(
    metric_name: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track custom metrics."""
    try:
        # For now, just log custom metrics
        logger.info(f"Custom metric: {metric_name}={value}, metadata={metadata}")
        
        return {"status": "success", "metric_name": metric_name}
    
    except Exception as e:
        logger.error(f"Error tracking custom metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/current")
async def get_current_metrics():
    """Get current aggregated metrics."""
    # Get current system metrics
    system_metrics = metrics_collector.get_current_system_metrics()
    
    # Get performance summary
    performance_summary = metrics_collector.get_performance_summary("1h")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": system_metrics.model_dump() if system_metrics else None,
        "performance": performance_summary.model_dump(),
        "status": "healthy"
    }


@app.get("/metrics/history")
async def get_metrics_history(
    metric_type: str = Query(..., description="Type of metrics to retrieve"),
    hours: int = Query(24, description="Hours of history to retrieve"),
    model_name: Optional[str] = Query(None, description="Filter by model name")
):
    """Get historical metrics."""
    if metric_type == "system":
        metrics = database_manager.get_system_metrics_history(hours)
        return [m.model_dump() for m in metrics]
    elif metric_type == "inference":
        metrics = database_manager.get_inference_metrics_history(hours, model_name)
        return [m.model_dump() for m in metrics]
    elif metric_type == "error":
        metrics = database_manager.get_error_metrics_history(hours)
        return [m.model_dump() for m in metrics]
    else:
        raise HTTPException(status_code=400, detail="Invalid metric type")


@app.get("/metrics/summary")
async def get_performance_summary(time_period: str = Query("1h", description="Time period for summary")):
    """Get performance summary."""
    summary = metrics_collector.get_performance_summary(time_period)
    return summary.model_dump()


@app.get("/alerts")
async def get_alerts(resolved: Optional[bool] = Query(None, description="Filter by resolved status")):
    """Get alerts."""
    alerts = alert_manager.get_alerts(resolved)
    return [alert.model_dump() for alert in alerts]


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    try:
        success = alert_manager.resolve_alert(alert_id)
        if success:
            return {"status": "success", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get database and system statistics."""
    try:
        db_stats = database_manager.get_database_stats()
        uptime = metrics_collector.get_uptime()
        
        return {
            "uptime_seconds": uptime,
            "database": db_stats,
            "active_websockets": len(connection_manager.active_connections)
        }
    
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


@app.get("/models")
async def get_models():
    """Get information about monitored models."""
    try:
        # Get unique model names from recent inference metrics
        recent_inferences = metrics_collector.get_recent_inference_metrics(1000)
        model_names = set(m.model_name for m in recent_inferences if m.model_name)
        
        models = []
        for model_name in model_names:
            model_inferences = [m for m in recent_inferences if m.model_name == model_name]
            if model_inferences:
                latest = model_inferences[0]
                models.append(ModelInfo(
                    name=model_name,
                    version=latest.model_version,
                    loaded_timestamp=min(m.timestamp for m in model_inferences),
                    metadata={
                        "total_requests": len(model_inferences),
                        "avg_response_time": sum(m.response_time_ms for m in model_inferences) / len(model_inferences),
                        "success_rate": sum(1 for m in model_inferences if m.success) / len(model_inferences) * 100
                    }
                ).model_dump())
        
        return models
    
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    """Run the monitoring server."""
    uvicorn.run(
        "api.server:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level=config.api.log_level
    )


if __name__ == "__main__":
    run_server() 