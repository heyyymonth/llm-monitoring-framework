import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, validator


class AlertThresholds(BaseModel):
    cpu_percent: float = 80.0
    memory_percent: float = 85.0
    response_time_ms: float = 5000.0
    gpu_memory_percent: float = 90.0
    error_rate_percent: float = 5.0


class MonitoringConfig(BaseModel):
    metrics_interval: float = 1.0
    max_history_days: int = 30
    alert_thresholds: AlertThresholds = AlertThresholds()


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    update_interval: int = 1000
    max_chart_points: int = 1000


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"


class DatabaseConfig(BaseModel):
    sqlite_path: str = "data/monitoring.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/monitoring.log"
    max_size_mb: int = 100
    backup_count: int = 5


class EmailConfig(BaseModel):
    enabled: bool = False
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    to_addresses: list = []


class AlertsConfig(BaseModel):
    enabled: bool = True
    webhook_url: Optional[str] = None
    email: EmailConfig = EmailConfig()


class Config(BaseModel):
    monitoring: MonitoringConfig = MonitoringConfig()
    dashboard: DashboardConfig = DashboardConfig()
    api: APIConfig = APIConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    alerts: AlertsConfig = AlertsConfig()


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file with environment variable overrides."""
    
    # Create default config
    config_data = {}
    
    # Load from file if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file) or {}
    
    # Override with environment variables
    env_overrides = {
        'api': {
            'host': os.getenv('API_HOST'),
            'port': int(os.getenv('API_PORT', 8000)),
        },
        'dashboard': {
            'host': os.getenv('DASHBOARD_HOST'),
            'port': int(os.getenv('DASHBOARD_PORT', 8080)),
        },
        'database': {
            'redis_host': os.getenv('REDIS_HOST'),
            'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        }
    }
    
    # Merge environment overrides
    for section, values in env_overrides.items():
        if section not in config_data:
            config_data[section] = {}
        for key, value in values.items():
            if value is not None:
                config_data[section][key] = value
    
    return Config(**config_data)


def ensure_directories(config: Config):
    """Ensure required directories exist."""
    
    # Create data directory
    data_dir = Path(config.database.sqlite_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    log_dir = Path(config.logging.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
        ensure_directories(_config)
    return _config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _config
    _config = config
    ensure_directories(_config) 