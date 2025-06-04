import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
import warnings

# Suppress the specific FutureWarning about datetime
warnings.filterwarnings("ignore", message=".*DatetimeProperties.to_pydatetime.*", category=FutureWarning)

from monitoring.config import get_config

logger = logging.getLogger(__name__)

# Global data storage
current_metrics = {}
metrics_history = []
inference_history = []
alerts_data = []
models_data = []

config = get_config()
API_BASE_URL = f"http://{config.api.host}:{config.api.port}"

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

app.title = "LLM Performance Monitor"

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🤖 LLM Performance Monitor", className="header-title"),
        html.P("Real-time monitoring of Large Language Model performance and actions", 
               className="header-subtitle"),
        html.Div([
            html.Div(id="status-indicator", className="status-indicator"),
            html.Span("System Status", className="status-label")
        ], className="status-container")
    ], className="header"),
    
    # Refresh interval
    dcc.Interval(
        id='interval-component',
        interval=config.dashboard.update_interval,  # Update every second
        n_intervals=0
    ),
    
    # Main content
    html.Div([
        # Top row - Key metrics cards
        html.Div([
            html.Div([
                html.H3("📊 System Overview", className="card-title"),
                html.Div(id="metrics-cards")
            ], className="card", style={'width': '100%'})
        ], className="row"),
        
        # Second row - Real-time charts
        html.Div([
            html.Div([
                html.H3("📈 System Metrics", className="card-title"),
                dcc.Graph(id="system-metrics-chart")
            ], className="card", style={'width': '50%'}),
            
            html.Div([
                html.H3("⚡ Inference Performance", className="card-title"),
                dcc.Graph(id="inference-metrics-chart")
            ], className="card", style={'width': '50%'})
        ], className="row"),
        
        # Third row - Detailed metrics
        html.Div([
            html.Div([
                html.H3("🎯 Response Time Distribution", className="card-title"),
                dcc.Graph(id="response-time-histogram")
            ], className="card", style={'width': '50%'}),
            
            html.Div([
                html.H3("🔥 Throughput & Tokens", className="card-title"),
                dcc.Graph(id="throughput-chart")
            ], className="card", style={'width': '50%'})
        ], className="row"),
        
        # Fourth row - Alerts and Models
        html.Div([
            html.Div([
                html.H3("🚨 Active Alerts", className="card-title"),
                html.Div(id="alerts-table")
            ], className="card", style={'width': '50%'}),
            
            html.Div([
                html.H3("🤖 Models", className="card-title"),
                html.Div(id="models-table")
            ], className="card", style={'width': '50%'})
        ], className="row"),
        
        # Fifth row - Error tracking
        html.Div([
            html.Div([
                html.H3("❌ Error Analysis", className="card-title"),
                dcc.Graph(id="error-chart")
            ], className="card", style={'width': '100%'})
        ], className="row")
        
    ], className="main-content")
], className="dashboard-container")


def fetch_data():
    """Fetch data from the API."""
    global current_metrics, metrics_history, inference_history, alerts_data, models_data
    
    try:
        # Get current metrics
        response = requests.get(f"{API_BASE_URL}/metrics/current", timeout=5)
        if response.status_code == 200:
            current_metrics = response.json()
        
        # Get system metrics history
        response = requests.get(f"{API_BASE_URL}/metrics/history?metric_type=system&hours=1", timeout=5)
        if response.status_code == 200:
            metrics_history = response.json()
        
        # Get inference metrics history
        response = requests.get(f"{API_BASE_URL}/metrics/history?metric_type=inference&hours=1", timeout=5)
        if response.status_code == 200:
            inference_history = response.json()
        
        # Get alerts
        response = requests.get(f"{API_BASE_URL}/alerts?resolved=false", timeout=5)
        if response.status_code == 200:
            alerts_data = response.json()
        
        # Get models
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            
    except Exception as e:
        logger.warning(f"Failed to fetch data from API: {e}")


def start_data_fetcher():
    """Start background thread to fetch data."""
    def fetch_loop():
        while True:
            fetch_data()
            time.sleep(5)  # Fetch every 5 seconds
    
    thread = threading.Thread(target=fetch_loop, daemon=True)
    thread.start()


@callback(Output('metrics-cards', 'children'), [Input('interval-component', 'n_intervals')])
def update_metrics_cards(n):
    """Update the metrics cards."""
    if not current_metrics:
        return html.P("No data available")
    
    system = current_metrics.get('system', {})
    performance = current_metrics.get('performance', {})
    
    cards = html.Div([
        # CPU Card
        html.Div([
            html.Div([
                html.I(className="fas fa-microchip"),
                html.H4(f"{system.get('cpu_percent', 0):.1f}%"),
                html.P("CPU Usage")
            ], className="metric-card cpu-card")
        ], className="metric-container"),
        
        # Memory Card
        html.Div([
            html.Div([
                html.I(className="fas fa-memory"),
                html.H4(f"{system.get('memory_percent', 0):.1f}%"),
                html.P("Memory Usage")
            ], className="metric-card memory-card")
        ], className="metric-container"),
        
        # Response Time Card
        html.Div([
            html.Div([
                html.I(className="fas fa-clock"),
                html.H4(f"{performance.get('avg_response_time_ms', 0):.0f}ms"),
                html.P("Avg Response Time")
            ], className="metric-card response-card")
        ], className="metric-container"),
        
        # Requests Card
        html.Div([
            html.Div([
                html.I(className="fas fa-chart-line"),
                html.H4(f"{performance.get('total_requests', 0)}"),
                html.P("Total Requests")
            ], className="metric-card requests-card")
        ], className="metric-container"),
        
        # Error Rate Card
        html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle"),
                html.H4(f"{performance.get('error_rate', 0):.1f}%"),
                html.P("Error Rate")
            ], className="metric-card error-card")
        ], className="metric-container"),
        
        # Tokens/sec Card
        html.Div([
            html.Div([
                html.I(className="fas fa-tachometer-alt"),
                html.H4(f"{performance.get('avg_tokens_per_second', 0):.1f}"),
                html.P("Tokens/sec")
            ], className="metric-card tokens-card")
        ], className="metric-container")
        
    ], className="metrics-grid")
    
    return cards


@callback(Output('system-metrics-chart', 'figure'), [Input('interval-component', 'n_intervals')])
def update_system_chart(n):
    """Update system metrics chart."""
    if not metrics_history:
        return go.Figure()
    
    df = pd.DataFrame(metrics_history)
    # Convert timestamp and ensure it's properly handled
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Convert datetime to proper format for plotly - fix deprecation warning
    timestamps = np.array(df['timestamp'].dt.to_pydatetime())
    
    # CPU usage
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df['cpu_percent'],
        mode='lines',
        name='CPU %',
        line=dict(color='#ff6b6b', width=2)
    ))
    
    # Memory usage
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df['memory_percent'],
        mode='lines',
        name='Memory %',
        line=dict(color='#4ecdc4', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="System Resource Usage",
        xaxis_title="Time",
        yaxis=dict(title="CPU %", side="left", color='#ff6b6b'),
        yaxis2=dict(title="Memory %", side="right", overlaying="y", color='#4ecdc4'),
        legend=dict(x=0, y=1),
        template="plotly_white",
        height=300
    )
    
    return fig


@callback(Output('inference-metrics-chart', 'figure'), [Input('interval-component', 'n_intervals')])
def update_inference_chart(n):
    """Update inference metrics chart."""
    if not inference_history:
        return go.Figure()
    
    df = pd.DataFrame(inference_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by minute and calculate averages - fix datetime handling
    df_resampled = df.set_index('timestamp').resample('1T').agg({
        'response_time_ms': 'mean',
        'tokens_per_second': 'mean',
        'success': 'count'
    })
    
    # Reset index and properly convert timestamps
    df_grouped = df_resampled.reset_index()
    timestamps = np.array(df_grouped['timestamp'].dt.to_pydatetime())
    
    fig = go.Figure()
    
    # Response time
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df_grouped['response_time_ms'],
        mode='lines+markers',
        name='Response Time (ms)',
        line=dict(color='#45b7d1', width=2)
    ))
    
    # Tokens per second (on secondary y-axis)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df_grouped['tokens_per_second'],
        mode='lines+markers',
        name='Tokens/sec',
        line=dict(color='#96ceb4', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Inference Performance",
        xaxis_title="Time",
        yaxis=dict(title="Response Time (ms)", side="left", color='#45b7d1'),
        yaxis2=dict(title="Tokens/sec", side="right", overlaying="y", color='#96ceb4'),
        legend=dict(x=0, y=1),
        template="plotly_white",
        height=300
    )
    
    return fig


@callback(Output('response-time-histogram', 'figure'), [Input('interval-component', 'n_intervals')])
def update_response_time_histogram(n):
    """Update response time histogram."""
    if not inference_history:
        return go.Figure()
    
    df = pd.DataFrame(inference_history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['response_time_ms'],
        nbinsx=30,
        name='Response Time Distribution',
        marker_color='#feca57'
    ))
    
    # Add percentile lines
    p50 = df['response_time_ms'].quantile(0.5)
    p95 = df['response_time_ms'].quantile(0.95)
    p99 = df['response_time_ms'].quantile(0.99)
    
    fig.add_vline(x=p50, line_dash="dash", line_color="green", 
                  annotation_text=f"P50: {p50:.0f}ms")
    fig.add_vline(x=p95, line_dash="dash", line_color="orange", 
                  annotation_text=f"P95: {p95:.0f}ms")
    fig.add_vline(x=p99, line_dash="dash", line_color="red", 
                  annotation_text=f"P99: {p99:.0f}ms")
    
    fig.update_layout(
        title="Response Time Distribution",
        xaxis_title="Response Time (ms)",
        yaxis_title="Count",
        template="plotly_white",
        height=300
    )
    
    return fig


@callback(Output('throughput-chart', 'figure'), [Input('interval-component', 'n_intervals')])
def update_throughput_chart(n):
    """Update throughput chart."""
    if not inference_history:
        return go.Figure()
    
    df = pd.DataFrame(inference_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by minute - fix datetime handling
    df_resampled = df.set_index('timestamp').resample('1T').agg({
        'total_tokens': 'sum',
        'success': 'count'
    })
    
    # Reset index and properly convert timestamps
    df_grouped = df_resampled.reset_index()
    timestamps = np.array(df_grouped['timestamp'].dt.to_pydatetime())
    
    fig = go.Figure()
    
    # Requests per minute
    fig.add_trace(go.Bar(
        x=timestamps,
        y=df_grouped['success'],
        name='Requests/min',
        marker_color='#a8e6cf',
        yaxis='y'
    ))
    
    # Tokens per minute
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=df_grouped['total_tokens'],
        mode='lines+markers',
        name='Tokens/min',
        line=dict(color='#ff8b94', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Throughput Metrics",
        xaxis_title="Time",
        yaxis=dict(title="Requests/min", side="left", color='#a8e6cf'),
        yaxis2=dict(title="Tokens/min", side="right", overlaying="y", color='#ff8b94'),
        legend=dict(x=0, y=1),
        template="plotly_white",
        height=300
    )
    
    return fig


@callback(Output('alerts-table', 'children'), [Input('interval-component', 'n_intervals')])
def update_alerts_table(n):
    """Update alerts table."""
    if not alerts_data:
        return html.P("No active alerts", className="no-data")
    
    # Create alerts cards
    alerts_cards = []
    for alert in alerts_data[:5]:  # Show only first 5 alerts
        severity_class = f"alert-{alert['severity']}"
        
        card = html.Div([
            html.Div([
                html.I(className=f"fas fa-exclamation-{alert['severity']}"),
                html.H5(alert['rule_name']),
                html.P(alert['message']),
                html.Small(f"Triggered: {alert['timestamp'][:19]}")
            ])
        ], className=f"alert-card {severity_class}")
        
        alerts_cards.append(card)
    
    return html.Div(alerts_cards)


@callback(Output('models-table', 'children'), [Input('interval-component', 'n_intervals')])
def update_models_table(n):
    """Update models table."""
    if not models_data:
        return html.P("No models detected", className="no-data")
    
    # Create model cards
    model_cards = []
    for model in models_data:
        metadata = model.get('metadata', {})
        
        card = html.Div([
            html.H5(model['name']),
            html.P(f"Requests: {metadata.get('total_requests', 0)}"),
            html.P(f"Avg Response: {metadata.get('avg_response_time', 0):.0f}ms"),
            html.P(f"Success Rate: {metadata.get('success_rate', 0):.1f}%")
        ], className="model-card")
        
        model_cards.append(card)
    
    return html.Div(model_cards)


@callback(Output('error-chart', 'figure'), [Input('interval-component', 'n_intervals')])
def update_error_chart(n):
    """Update error analysis chart."""
    if not inference_history:
        return go.Figure()
    
    df = pd.DataFrame(inference_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by hour and calculate error rates
    df_grouped = df.set_index('timestamp').resample('1H').agg({
        'success': ['count', 'sum']
    }).reset_index()
    
    df_grouped.columns = ['timestamp', 'total_requests', 'successful_requests']
    df_grouped['error_rate'] = ((df_grouped['total_requests'] - df_grouped['successful_requests']) / 
                               df_grouped['total_requests'] * 100).fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_grouped['timestamp'],
        y=df_grouped['error_rate'],
        name='Error Rate %',
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title="Error Rate Over Time",
        xaxis_title="Time",
        yaxis_title="Error Rate (%)",
        template="plotly_white",
        height=300
    )
    
    return fig


@callback(Output('status-indicator', 'className'), [Input('interval-component', 'n_intervals')])
def update_status_indicator(n):
    """Update system status indicator."""
    if not current_metrics:
        return "status-indicator status-unknown"
    
    system = current_metrics.get('system', {})
    performance = current_metrics.get('performance', {})
    
    cpu_percent = system.get('cpu_percent', 0)
    memory_percent = system.get('memory_percent', 0)
    error_rate = performance.get('error_rate', 0)
    
    if cpu_percent > 90 or memory_percent > 95 or error_rate > 25:
        return "status-indicator status-critical"
    elif cpu_percent > 80 or memory_percent > 85 or error_rate > 10:
        return "status-indicator status-warning"
    else:
        return "status-indicator status-healthy"


# Add CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        .dashboard-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header-title {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .header-subtitle {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        
        .status-container {
            margin-top: 15px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-critical { background-color: #dc3545; }
        .status-unknown { background-color: #6c757d; }
        
        .row {
            display: flex;
            margin-bottom: 20px;
            gap: 20px;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .card-title {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.2em;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            color: white;
        }
        
        .metric-card i {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .metric-card h4 {
            margin: 10px 0 5px 0;
            font-size: 1.8em;
        }
        
        .metric-card p {
            margin: 0;
            opacity: 0.9;
        }
        
        .cpu-card { background: linear-gradient(135deg, #ff6b6b, #ee5a24); }
        .memory-card { background: linear-gradient(135deg, #4ecdc4, #44a08d); }
        .response-card { background: linear-gradient(135deg, #45b7d1, #1e3c72); }
        .requests-card { background: linear-gradient(135deg, #96ceb4, #659999); }
        .error-card { background: linear-gradient(135deg, #feca57, #ff9ff3); }
        .tokens-card { background: linear-gradient(135deg, #ff8b94, #ff6b6b); }
        
        .alert-card {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }
        
        .alert-error {
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }
        
        .alert-critical {
            background-color: #f8d7da;
            border-left-color: #721c24;
        }
        
        .model-card {
            padding: 15px;
            margin-bottom: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .no-data {
            text-align: center;
            color: #6c757d;
            font-style: italic;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''


def run_dashboard():
    """Run the dashboard server."""
    # Start data fetcher
    start_data_fetcher()
    
    # Run the app
    app.run_server(
        host=config.dashboard.host,
        port=config.dashboard.port,
        debug=False
    )


if __name__ == "__main__":
    run_dashboard() 