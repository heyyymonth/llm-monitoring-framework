import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
UPDATE_INTERVAL = 5000  # milliseconds

app = dash.Dash(__name__)

def get_health_data():
    """Fetch health data from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching health data: {e}")
    return None

def get_metrics_data():
    """Fetch current metrics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/current", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching metrics data: {e}")
    return None

def create_status_cards(health_data):
    """Create status cards for key LLM metrics."""
    if not health_data:
        return html.Div("No data available", className="alert alert-danger")
    
    system_metrics = health_data.get('system_metrics', {})
    
    # Calculate status colors
    def get_color(value, warning_threshold, critical_threshold):
        if value > critical_threshold:
            return "#dc3545"  # Red
        elif value > warning_threshold:
            return "#ffc107"  # Yellow
        else:
            return "#28a745"  # Green
    
    cpu_color = get_color(system_metrics.get('cpu_percent', 0), 70, 90)
    memory_color = get_color(system_metrics.get('memory_percent', 0), 75, 90)
    temp_color = get_color(system_metrics.get('cpu_temp_celsius', 0), 70, 85)
    
    cards = [
        # CPU Usage
        html.Div([
            html.H5("CPU Usage", className="card-title"),
            html.H3(f"{system_metrics.get('cpu_percent', 0):.1f}%", 
                    style={'color': cpu_color}),
            html.P("Current CPU utilization")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Memory Usage
        html.Div([
            html.H5("Memory Usage", className="card-title"),
            html.H3(f"{system_metrics.get('memory_percent', 0):.1f}%", 
                    style={'color': memory_color}),
            html.P(f"{system_metrics.get('memory_used_gb', 0):.1f}GB / "
                   f"{system_metrics.get('memory_total_gb', 0):.1f}GB")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Available Memory for Model Loading
        html.Div([
            html.H5("Available Memory", className="card-title"),
            html.H3(f"{system_metrics.get('available_memory_gb', 0):.1f}GB"),
            html.P("Free for model loading")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # GPU Count
        html.Div([
            html.H5("GPU Count", className="card-title"),
            html.H3(f"{system_metrics.get('gpu_count', 0)}"),
            html.P("Available GPUs")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # CPU Temperature
        html.Div([
            html.H5("CPU Temperature", className="card-title"),
            html.H3(f"{system_metrics.get('cpu_temp_celsius', 0):.1f}°C", 
                    style={'color': temp_color}),
            html.P("CPU thermal status")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Inference Threads
        html.Div([
            html.H5("Inference Threads", className="card-title"),
            html.H3(f"{system_metrics.get('llm_process_metrics', {}).get('inference_threads', 0)}"),
            html.P("Active LLM threads")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
    ]
    
    return html.Div(cards, className="row")

def create_llm_performance_charts(health_data):
    """Create LLM performance charts."""
    if not health_data:
        return html.Div("No data available")
    
    system_metrics = health_data.get('system_metrics', {})
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU & Memory Usage', 'GPU Utilization', 'Memory Breakdown', 'Thermal Status'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "indicator"}]]
    )
    
    # CPU & Memory Usage over time (simplified with current values)
    fig.add_trace(
        go.Scatter(x=[datetime.now()], y=[system_metrics.get('cpu_percent', 0)], 
                  name='CPU %', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[datetime.now()], y=[system_metrics.get('memory_percent', 0)], 
                  name='Memory %', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # GPU Utilization
    gpu_metrics = system_metrics.get('gpu_metrics', [])
    if gpu_metrics:
        gpu_names = [f"GPU {gpu.get('gpu_id', i)}" for i, gpu in enumerate(gpu_metrics)]
        gpu_utils = [gpu.get('utilization_percent', 0) for gpu in gpu_metrics]
        
        fig.add_trace(
            go.Bar(x=gpu_names, y=gpu_utils, name='GPU Utilization %',
                  marker=dict(color='green')),
            row=1, col=2
        )
    
    # Memory Breakdown
    memory_used = system_metrics.get('memory_used_gb', 0)
    memory_available = system_metrics.get('available_memory_gb', 0)
    
    fig.add_trace(
        go.Pie(labels=['Used', 'Available'], 
               values=[memory_used, memory_available],
               name="Memory"),
        row=2, col=1
    )
    
    # Thermal Status Indicator
    cpu_temp = system_metrics.get('cpu_temp_celsius', 0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=cpu_temp,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU Temp (°C)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 60], 'color': "lightgray"},
                       {'range': [60, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 85}}),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    return dcc.Graph(figure=fig)

def create_llm_process_charts(health_data):
    """Create LLM process-specific charts."""
    if not health_data:
        return html.Div("No data available")
    
    llm_process = health_data.get('system_metrics', {}).get('llm_process_metrics', {})
    
    if not llm_process:
        return html.Div("LLM process metrics not available", className="alert alert-warning")
    
    # Create process metrics display
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('LLM Process Memory', 'Process Info'),
        specs=[[{"type": "bar"}, {"type": "table"}]]
    )
    
    # Memory breakdown
    memory_labels = ['RSS Memory', 'Model Memory']
    memory_values = [
        llm_process.get('memory_rss_mb', 0),
        llm_process.get('model_memory_mb', 0)
    ]
    
    fig.add_trace(
        go.Bar(x=memory_labels, y=memory_values, name='Memory (MB)',
               marker=dict(color=['skyblue', 'orange'])),
        row=1, col=1
    )
    
    # Process info table
    process_info = [
        ['Process ID', str(llm_process.get('pid', 'N/A'))],
        ['CPU Usage', f"{llm_process.get('cpu_percent', 0):.1f}%"],
        ['Memory %', f"{llm_process.get('memory_percent', 0):.1f}%"],
        ['Inference Threads', str(llm_process.get('inference_threads', 0))],
        ['RSS Memory', f"{llm_process.get('memory_rss_mb', 0):.1f} MB"],
        ['Model Memory', f"{llm_process.get('model_memory_mb', 0):.1f} MB"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='lightgray',
                       align='left'),
            cells=dict(values=list(zip(*process_info)),
                      fill_color='white',
                      align='left')),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return dcc.Graph(figure=fig)

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("LLM Performance Monitor", className="text-center mb-4"),
        html.P("Real-time monitoring of LLM inference performance", 
               className="text-center text-muted")
    ], className="container mt-3"),
    
    # Status Cards
    html.Div([
        html.H3("System Status", className="mb-3"),
        html.Div(id="status-cards")
    ], className="container mb-4"),
    
    # Main Charts
    html.Div([
        html.H3("LLM Performance Metrics", className="mb-3"),
        html.Div(id="llm-performance-charts")
    ], className="container mb-4"),
    
    # Process Charts
    html.Div([
        html.H3("LLM Process Details", className="mb-3"),
        html.Div(id="llm-process-charts")
    ], className="container mb-4"),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=UPDATE_INTERVAL,
        n_intervals=0
    ),
    
    # Bootstrap CSS
    html.Link(
        rel='stylesheet',
        href='https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
    )
])

# Callbacks
@app.callback(
    [Output('status-cards', 'children'),
     Output('llm-performance-charts', 'children'),
     Output('llm-process-charts', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components."""
    health_data = get_health_data()
    
    status_cards = create_status_cards(health_data)
    performance_charts = create_llm_performance_charts(health_data)
    process_charts = create_llm_process_charts(health_data)
    
    return status_cards, performance_charts, process_charts

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080) 