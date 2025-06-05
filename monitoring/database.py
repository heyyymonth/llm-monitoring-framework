import asyncio
import json
import sqlite3
import redis
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import threading
import logging

from .models import (
    SystemMetrics, InferenceMetrics, ErrorMetrics, 
    QueueMetrics, PerformanceSummary, Alert
)
from .config import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages data storage in SQLite and Redis."""
    
    def __init__(self):
        self.config = get_config()
        self._lock = threading.Lock()
        self._redis_client: Optional[redis.Redis] = None
        self._setup_sqlite()
        self._setup_redis()
    
    def _setup_sqlite(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.config.database.sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_gb REAL NOT NULL,
                    memory_total_gb REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    gpu_count INTEGER DEFAULT 0,
                    gpu_metrics TEXT,
                    network_bytes_sent INTEGER DEFAULT 0,
                    network_bytes_recv INTEGER DEFAULT 0,
                    load_average TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    request_id TEXT NOT NULL,
                    model_name TEXT,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time_ms REAL NOT NULL,
                    queue_time_ms REAL DEFAULT 0,
                    processing_time_ms REAL DEFAULT 0,
                    tokens_per_second REAL DEFAULT 0,
                    prompt_length INTEGER DEFAULT 0,
                    response_length INTEGER DEFAULT 0,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    model_version TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    request_id TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT,
                    model_name TEXT,
                    endpoint TEXT,
                    user_id TEXT,
                    severity TEXT DEFAULT 'error'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    pending_requests INTEGER NOT NULL,
                    processing_requests INTEGER NOT NULL,
                    completed_requests INTEGER NOT NULL,
                    failed_requests INTEGER NOT NULL,
                    avg_wait_time_ms REAL NOT NULL,
                    max_wait_time_ms REAL NOT NULL,
                    queue_throughput REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_timestamp DATETIME,
                    metadata TEXT
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_inference_model ON inference_metrics(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_error_timestamp ON error_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_timestamp ON queue_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved)")
            
            conn.commit()
        
        logger.info(f"SQLite database initialized at {self.config.database.sqlite_path}")
    
    def _setup_redis(self):
        """Initialize Redis connection."""
        try:
            self._redis_client = redis.Redis(
                host=self.config.database.redis_host,
                port=self.config.database.redis_port,
                db=self.config.database.redis_db,
                decode_responses=True
            )
            # Test connection
            self._redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Continuing without Redis cache.")
            self._redis_client = None
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in SQLite."""
        with self._lock:
            with sqlite3.connect(self.config.database.sqlite_path) as conn:
                                # Convert enhanced metrics to simplified format for compatibility
                disk_percent = 0.0  # Default fallback
                network_bytes_sent = 0
                network_bytes_recv = 0
                
                # Extract network totals from enhanced metrics
                if metrics.network_metrics:
                    network_bytes_sent = sum(nm.bytes_sent_per_sec for nm in metrics.network_metrics if hasattr(nm, 'bytes_sent_per_sec'))
                    network_bytes_recv = sum(nm.bytes_recv_per_sec for nm in metrics.network_metrics if hasattr(nm, 'bytes_recv_per_sec'))
                
                # Create simplified load average from enhanced scheduler metrics
                load_average = [
                    getattr(metrics, 'system_load_1m', 0.0),
                    getattr(metrics, 'system_load_5m', 0.0),
                    getattr(metrics, 'system_load_15m', 0.0)
                ]
                
                conn.execute("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, memory_used_gb,
                        memory_total_gb, disk_percent, gpu_count, gpu_metrics,
                        network_bytes_sent, network_bytes_recv, load_average
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_used_gb,
                    metrics.memory_total_gb,
                    disk_percent,  # Use fallback value
                    metrics.gpu_count,
                    json.dumps(metrics.gpu_metrics),
                    network_bytes_sent,
                    network_bytes_recv,
                    json.dumps(load_average)
                ))
                conn.commit()
        
        # Store in Redis for fast access
        if self._redis_client:
            try:
                self._redis_client.set(
                    "system:current",
                    metrics.json(),
                    ex=300  # Expire after 5 minutes
                )
            except Exception as e:
                logger.warning(f"Failed to store system metrics in Redis: {e}")
    
    def store_inference_metrics(self, metrics: InferenceMetrics):
        """Store inference metrics in SQLite."""
        with self._lock:
            with sqlite3.connect(self.config.database.sqlite_path) as conn:
                conn.execute("""
                    INSERT INTO inference_metrics (
                        timestamp, request_id, model_name, prompt_tokens, completion_tokens,
                        total_tokens, response_time_ms, queue_time_ms, processing_time_ms,
                        tokens_per_second, prompt_length, response_length, success,
                        error_message, model_version, temperature, max_tokens, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.request_id,
                    metrics.model_name,
                    metrics.prompt_tokens,
                    metrics.completion_tokens,
                    metrics.total_tokens,
                    metrics.response_time_ms,
                    metrics.queue_time_ms,
                    metrics.processing_time_ms,
                    metrics.tokens_per_second,
                    metrics.prompt_length,
                    metrics.response_length,
                    metrics.success,
                    metrics.error_message,
                    metrics.model_version,
                    metrics.temperature,
                    metrics.max_tokens,
                    json.dumps(metrics.metadata)
                ))
                conn.commit()
        
        # Store recent metrics in Redis
        if self._redis_client:
            try:
                self._redis_client.lpush("inference:recent", metrics.json())
                self._redis_client.ltrim("inference:recent", 0, 999)  # Keep last 1000
            except Exception as e:
                logger.warning(f"Failed to store inference metrics in Redis: {e}")
    
    def store_error_metrics(self, metrics: ErrorMetrics):
        """Store error metrics in SQLite."""
        with self._lock:
            with sqlite3.connect(self.config.database.sqlite_path) as conn:
                conn.execute("""
                    INSERT INTO error_metrics (
                        timestamp, request_id, error_type, error_message, stack_trace,
                        model_name, endpoint, user_id, severity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.request_id,
                    metrics.error_type,
                    metrics.error_message,
                    metrics.stack_trace,
                    metrics.model_name,
                    metrics.endpoint,
                    metrics.user_id,
                    metrics.severity.value
                ))
                conn.commit()
        
        # Store in Redis for alerts
        if self._redis_client:
            try:
                self._redis_client.lpush("errors:recent", metrics.json())
                self._redis_client.ltrim("errors:recent", 0, 99)  # Keep last 100
            except Exception as e:
                logger.warning(f"Failed to store error metrics in Redis: {e}")
    
    def store_queue_metrics(self, metrics: QueueMetrics):
        """Store queue metrics in SQLite."""
        with self._lock:
            with sqlite3.connect(self.config.database.sqlite_path) as conn:
                conn.execute("""
                    INSERT INTO queue_metrics (
                        timestamp, pending_requests, processing_requests, completed_requests,
                        failed_requests, avg_wait_time_ms, max_wait_time_ms, queue_throughput
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.pending_requests,
                    metrics.processing_requests,
                    metrics.completed_requests,
                    metrics.failed_requests,
                    metrics.avg_wait_time_ms,
                    metrics.max_wait_time_ms,
                    metrics.queue_throughput
                ))
                conn.commit()
    
    def store_alert(self, alert: Alert):
        """Store alert in SQLite."""
        with self._lock:
            with sqlite3.connect(self.config.database.sqlite_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts (
                        id, timestamp, rule_id, rule_name, severity, message,
                        metric_value, threshold, resolved, resolved_timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.timestamp,
                    alert.rule_id,
                    alert.rule_name,
                    alert.severity.value,
                    alert.message,
                    alert.metric_value,
                    alert.threshold,
                    alert.resolved,
                    alert.resolved_timestamp,
                    json.dumps(alert.metadata)
                ))
                conn.commit()
    
    def get_system_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics history from SQLite."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.config.database.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM system_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """, (cutoff,))
            
            metrics = []
            for row in cursor.fetchall():
                # Create compatible SystemMetrics from legacy database format
                load_avg = json.loads(row['load_average']) if row['load_average'] else [0.0, 0.0, 0.0]
                metrics.append(SystemMetrics(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    cpu_percent=row['cpu_percent'],
                    memory_percent=row['memory_percent'],
                    memory_used_gb=row['memory_used_gb'],
                    memory_total_gb=row['memory_total_gb'],
                    available_memory_gb=row['memory_total_gb'] - row['memory_used_gb'],  # Calculate available
                    memory_pressure=row['memory_percent'] > 85,  # Estimate pressure
                    system_load_1m=load_avg[0] if len(load_avg) > 0 else 0.0,
                    system_load_5m=load_avg[1] if len(load_avg) > 1 else 0.0,
                    system_load_15m=load_avg[2] if len(load_avg) > 2 else 0.0,
                    gpu_count=row['gpu_count'],
                    gpu_metrics=json.loads(row['gpu_metrics']) if row['gpu_metrics'] else [],
                    # Initialize enhanced metrics as empty lists for compatibility
                    disk_io_metrics=[],
                    network_metrics=[],
                    thermal_zones={}
                ))
            
            return metrics
    
    def get_inference_metrics_history(self, hours: int = 24, model_name: Optional[str] = None) -> List[InferenceMetrics]:
        """Get inference metrics history from SQLite."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        query = """
            SELECT * FROM inference_metrics 
            WHERE timestamp >= ?
        """
        params = [cutoff]
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.config.database.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append(InferenceMetrics(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    request_id=row['request_id'],
                    model_name=row['model_name'],
                    prompt_tokens=row['prompt_tokens'],
                    completion_tokens=row['completion_tokens'],
                    total_tokens=row['total_tokens'],
                    response_time_ms=row['response_time_ms'],
                    queue_time_ms=row['queue_time_ms'],
                    processing_time_ms=row['processing_time_ms'],
                    tokens_per_second=row['tokens_per_second'],
                    prompt_length=row['prompt_length'],
                    response_length=row['response_length'],
                    success=bool(row['success']),
                    error_message=row['error_message'],
                    model_version=row['model_version'],
                    temperature=row['temperature'],
                    max_tokens=row['max_tokens'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))
            
            return metrics
    
    def get_error_metrics_history(self, hours: int = 24) -> List[ErrorMetrics]:
        """Get error metrics history from SQLite."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.config.database.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM error_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """, (cutoff,))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append(ErrorMetrics(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    request_id=row['request_id'],
                    error_type=row['error_type'],
                    error_message=row['error_message'],
                    stack_trace=row['stack_trace'],
                    model_name=row['model_name'],
                    endpoint=row['endpoint'],
                    user_id=row['user_id'],
                    severity=row['severity']
                ))
            
            return metrics
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics from Redis or SQLite."""
        if self._redis_client:
            try:
                data = self._redis_client.get("system:current")
                if data:
                    return SystemMetrics.parse_raw(data)
            except Exception as e:
                logger.warning(f"Failed to get system metrics from Redis: {e}")
        
        # Fallback to SQLite
        with sqlite3.connect(self.config.database.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM system_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                # Create compatible SystemMetrics from legacy database format
                load_avg = json.loads(row['load_average']) if row['load_average'] else [0.0, 0.0, 0.0]
                return SystemMetrics(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    cpu_percent=row['cpu_percent'],
                    memory_percent=row['memory_percent'],
                    memory_used_gb=row['memory_used_gb'],
                    memory_total_gb=row['memory_total_gb'],
                    available_memory_gb=row['memory_total_gb'] - row['memory_used_gb'],  # Calculate available
                    memory_pressure=row['memory_percent'] > 85,  # Estimate pressure
                    system_load_1m=load_avg[0] if len(load_avg) > 0 else 0.0,
                    system_load_5m=load_avg[1] if len(load_avg) > 1 else 0.0,
                    system_load_15m=load_avg[2] if len(load_avg) > 2 else 0.0,
                    gpu_count=row['gpu_count'],
                    gpu_metrics=json.loads(row['gpu_metrics']) if row['gpu_metrics'] else [],
                    # Initialize enhanced metrics as empty lists for compatibility
                    disk_io_metrics=[],
                    network_metrics=[],
                    thermal_zones={}
                )
        
        return None
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from SQLite."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._lock:
            with sqlite3.connect(self.config.database.sqlite_path) as conn:
                tables = ['system_metrics', 'inference_metrics', 'error_metrics', 'queue_metrics']
                
                for table in tables:
                    result = conn.execute(f"""
                        DELETE FROM {table} 
                        WHERE timestamp < ?
                    """, (cutoff,))
                    
                    if result.rowcount > 0:
                        logger.info(f"Cleaned up {result.rowcount} old records from {table}")
                
                conn.commit()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        with sqlite3.connect(self.config.database.sqlite_path) as conn:
            tables = ['system_metrics', 'inference_metrics', 'error_metrics', 'queue_metrics', 'alerts']
            
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"{table}_count"] = count
        
        if self._redis_client:
            try:
                redis_info = self._redis_client.info()
                stats['redis_connected'] = True
                stats['redis_used_memory'] = redis_info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = self._redis_client.dbsize()
            except Exception as e:
                stats['redis_connected'] = False
                stats['redis_error'] = str(e)
        else:
            stats['redis_connected'] = False
        
        return stats 