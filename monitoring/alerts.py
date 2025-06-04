import threading
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from collections import defaultdict

from .models import (
    Alert, AlertRule, AlertLevel, MetricType, SystemMetrics, 
    InferenceMetrics, PerformanceSummary
)
from .config import get_config

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts based on metric thresholds."""
    
    def __init__(self, metrics_collector, database_manager):
        self.config = get_config()
        self.metrics_collector = metrics_collector
        self.database_manager = database_manager
        
        self._running = False
        self._alert_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Active alerts
        self._active_alerts: Dict[str, Alert] = {}
        
        # Alert rules
        self._alert_rules = self._create_default_rules()
        
        # Alert cooldowns (to prevent spam)
        self._alert_cooldowns: Dict[str, datetime] = {}
        self._cooldown_period = timedelta(minutes=5)
    
    def _create_default_rules(self) -> List[AlertRule]:
        """Create default alert rules based on configuration."""
        thresholds = self.config.monitoring.alert_thresholds
        
        rules = [
            AlertRule(
                id="cpu_high",
                name="High CPU Usage",
                metric_type=MetricType.SYSTEM,
                metric_name="cpu_percent",
                threshold=thresholds.cpu_percent,
                comparison="gte",
                severity=AlertLevel.WARNING,
                description=f"CPU usage is above {thresholds.cpu_percent}%"
            ),
            AlertRule(
                id="memory_high",
                name="High Memory Usage",
                metric_type=MetricType.SYSTEM,
                metric_name="memory_percent",
                threshold=thresholds.memory_percent,
                comparison="gte",
                severity=AlertLevel.WARNING,
                description=f"Memory usage is above {thresholds.memory_percent}%"
            ),
            AlertRule(
                id="response_time_high",
                name="High Response Time",
                metric_type=MetricType.INFERENCE,
                metric_name="avg_response_time_ms",
                threshold=thresholds.response_time_ms,
                comparison="gte",
                severity=AlertLevel.WARNING,
                description=f"Average response time is above {thresholds.response_time_ms}ms"
            ),
            AlertRule(
                id="error_rate_high",
                name="High Error Rate",
                metric_type=MetricType.INFERENCE,
                metric_name="error_rate",
                threshold=thresholds.error_rate_percent,
                comparison="gte",
                severity=AlertLevel.ERROR,
                description=f"Error rate is above {thresholds.error_rate_percent}%"
            ),
            AlertRule(
                id="gpu_memory_high",
                name="High GPU Memory Usage",
                metric_type=MetricType.SYSTEM,
                metric_name="gpu_memory_percent",
                threshold=thresholds.gpu_memory_percent,
                comparison="gte",
                severity=AlertLevel.WARNING,
                description=f"GPU memory usage is above {thresholds.gpu_memory_percent}%"
            ),
        ]
        
        return rules
    
    def start(self):
        """Start alert monitoring."""
        if not self._running:
            self._running = True
            self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
            self._alert_thread.start()
            logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert monitoring."""
        self._running = False
        if self._alert_thread:
            self._alert_thread.join()
        logger.info("Alert manager stopped")
    
    def _alert_loop(self):
        """Main alert checking loop."""
        while self._running:
            try:
                self._check_alerts()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_alerts(self):
        """Check all alert rules against current metrics."""
        current_metrics = self.metrics_collector.get_current_system_metrics()
        performance_summary = self.metrics_collector.get_performance_summary("1h")
        
        for rule in self._alert_rules:
            if not rule.enabled:
                continue
                
            try:
                should_alert, metric_value = self._evaluate_rule(
                    rule, current_metrics, performance_summary
                )
                
                if should_alert:
                    self._trigger_alert(rule, metric_value)
                else:
                    # Check if we should resolve an existing alert
                    self._check_alert_resolution(rule, metric_value)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.id}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, system_metrics: Optional[SystemMetrics], 
                      performance_summary: PerformanceSummary) -> tuple[bool, float]:
        """Evaluate an alert rule against current metrics."""
        metric_value = 0.0
        
        # Get metric value based on rule type and metric name
        if rule.metric_type == MetricType.SYSTEM and system_metrics:
            if rule.metric_name == "cpu_percent":
                metric_value = system_metrics.cpu_percent
            elif rule.metric_name == "memory_percent":
                metric_value = system_metrics.memory_percent
            elif rule.metric_name == "gpu_memory_percent":
                if system_metrics.gpu_metrics:
                    # Take average GPU memory usage
                    gpu_memories = [gpu['memory_percent'] for gpu in system_metrics.gpu_metrics]
                    metric_value = sum(gpu_memories) / len(gpu_memories)
                else:
                    metric_value = 0.0
            else:
                return False, 0.0
                
        elif rule.metric_type == MetricType.INFERENCE:
            if rule.metric_name == "avg_response_time_ms":
                metric_value = performance_summary.avg_response_time_ms
            elif rule.metric_name == "error_rate":
                metric_value = performance_summary.error_rate
            else:
                return False, 0.0
        else:
            return False, 0.0
        
        # Evaluate threshold
        should_alert = False
        if rule.comparison == "gte":
            should_alert = metric_value >= rule.threshold
        elif rule.comparison == "gt":
            should_alert = metric_value > rule.threshold
        elif rule.comparison == "lte":
            should_alert = metric_value <= rule.threshold
        elif rule.comparison == "lt":
            should_alert = metric_value < rule.threshold
        elif rule.comparison == "eq":
            should_alert = metric_value == rule.threshold
        
        return should_alert, metric_value
    
    def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """Trigger an alert for a rule."""
        alert_id = f"{rule.id}_{int(time.time())}"
        
        # Check cooldown
        if rule.id in self._alert_cooldowns:
            if datetime.utcnow() - self._alert_cooldowns[rule.id] < self._cooldown_period:
                return  # Still in cooldown
        
        # Create alert
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.description}. Current value: {metric_value:.2f}",
            metric_value=metric_value,
            threshold=rule.threshold,
            metadata={
                "rule_description": rule.description,
                "comparison": rule.comparison
            }
        )
        
        with self._lock:
            self._active_alerts[alert_id] = alert
            self._alert_cooldowns[rule.id] = datetime.utcnow()
        
        # Store in database
        self.database_manager.store_alert(alert)
        
        # Send notifications
        self._send_alert_notification(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
    
    def _check_alert_resolution(self, rule: AlertRule, metric_value: float):
        """Check if any alerts for this rule should be resolved."""
        alerts_to_resolve = []
        
        with self._lock:
            for alert_id, alert in self._active_alerts.items():
                if alert.rule_id == rule.id and not alert.resolved:
                    # Check if metric is now below threshold (for gte/gt comparisons)
                    if rule.comparison in ["gte", "gt"]:
                        if metric_value < rule.threshold * 0.9:  # 10% below threshold
                            alerts_to_resolve.append(alert_id)
                    elif rule.comparison in ["lte", "lt"]:
                        if metric_value > rule.threshold * 1.1:  # 10% above threshold
                            alerts_to_resolve.append(alert_id)
        
        # Resolve alerts
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id)
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification (webhook, email, etc.)."""
        try:
            # Webhook notification
            if self.config.alerts.webhook_url:
                # TODO: Implement webhook notification
                pass
            
            # Email notification
            if self.config.alerts.email.enabled:
                # TODO: Implement email notification
                pass
                
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = datetime.utcnow()
                
                # Update in database
                self.database_manager.store_alert(alert)
                
                logger.info(f"Alert resolved: {alert.rule_name}")
                return True
        
        return False
    
    def get_alerts(self, resolved: Optional[bool] = None) -> List[Alert]:
        """Get alerts, optionally filtered by resolved status."""
        with self._lock:
            alerts = list(self._active_alerts.values())
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self._alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        for i, rule in enumerate(self._alert_rules):
            if rule.id == rule_id:
                del self._alert_rules[i]
                logger.info(f"Removed alert rule: {rule.name}")
                return True
        return False
    
    def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return self._alert_rules.copy()
    
    def update_alert_rule(self, rule_id: str, **updates) -> bool:
        """Update an alert rule."""
        for rule in self._alert_rules:
            if rule.id == rule_id:
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logger.info(f"Updated alert rule: {rule.name}")
                return True
        return False 