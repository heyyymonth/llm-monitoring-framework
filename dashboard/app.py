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
            data = response.json()
            # Debug logging
            system = data.get('system', {})
            print(f"[DEBUG] Dashboard received: CPU {system.get('cpu_percent', 0)}%, Memory {system.get('memory_percent', 0)}%")
            return data
    except Exception as e:
        logger.error(f"Error fetching metrics data: {e}")
        print(f"[DEBUG] Error fetching metrics: {e}")
    return None

def create_status_cards(metrics_data):
    """Create status cards for key LLM metrics."""
    if not metrics_data:
        return html.Div("No data available", className="alert alert-danger")
    
    system_metrics = metrics_data.get('system', {})
    performance_metrics = metrics_data.get('performance', {})
    
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
            html.H3(f"{system_metrics.get('memory_available_gb', 0):.1f}GB"),
            html.P("Free for model loading")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Total Requests
        html.Div([
            html.H5("Total Requests", className="card-title"),
            html.H3(f"{performance_metrics.get('total_requests', 0)}"),
            html.P("LLM inference requests")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Success Rate
        html.Div([
            html.H5("Success Rate", className="card-title"),
            html.H3(f"{100 - performance_metrics.get('error_rate', 0):.1f}%", 
                    style={'color': '#28a745' if performance_metrics.get('error_rate', 0) < 5 else '#ffc107'}),
            html.P("Request success rate")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # LLM Process Memory
        html.Div([
            html.H5("LLM Process Memory", className="card-title"),
            html.H3(f"{system_metrics.get('llm_process', {}).get('memory_rss_mb', 0):.0f}MB"),
            html.P("LLM process RSS memory")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
    ]
    
    return html.Div(cards, className="row")

def create_llm_performance_charts(metrics_data):
    """Create LLM performance charts."""
    if not metrics_data:
        return html.Div("No data available")
    
    system_metrics = metrics_data.get('system', {})
    performance_metrics = metrics_data.get('performance', {})
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU & Memory Usage', 'Token Processing', 'Memory Breakdown', 'Response Time Performance'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]]
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
    
    # Token Processing Rate
    tokens_per_second = performance_metrics.get('avg_tokens_per_second', 0)
    total_tokens = performance_metrics.get('total_tokens_processed', 0)
    
    fig.add_trace(
        go.Bar(x=['Tokens/Second', 'Total Tokens'], 
               y=[tokens_per_second, total_tokens],
               name='Token Processing',
               marker=dict(color='green')),
        row=1, col=2
    )
    
    # Memory Breakdown
    memory_used = system_metrics.get('memory_used_gb', 0)
    memory_available = system_metrics.get('memory_available_gb', 0)
    
    fig.add_trace(
        go.Pie(labels=['Used', 'Available'], 
               values=[memory_used, memory_available],
               name="Memory"),
        row=2, col=1
    )
    
    # Response Time Performance
    avg_response_time = performance_metrics.get('avg_response_time_ms', 0)
    response_time_color = 'green' if avg_response_time < 500 else 'yellow' if avg_response_time < 1000 else 'red'
    
    fig.add_trace(
        go.Bar(x=['Avg Response Time', 'P95 Response Time'], 
               y=[performance_metrics.get('avg_response_time_ms', 0),
                  performance_metrics.get('p95_response_time_ms', 0)],
               name='Response Time (ms)',
               marker=dict(color=[response_time_color, 'orange'])),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    return dcc.Graph(figure=fig)

def create_llm_process_charts(metrics_data):
    """Create LLM process-specific charts."""
    if not metrics_data:
        return html.Div("No data available")
    
    llm_process = metrics_data.get('system', {}).get('llm_process', {})
    
    if not llm_process:
        return html.Div("LLM process metrics not available", className="alert alert-warning")
    
    # Create process metrics display
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('LLM Process Memory', 'Process Info'),
        specs=[[{"type": "bar"}, {"type": "table"}]]
    )
    
    # Memory breakdown
    memory_labels = ['RSS Memory']
    memory_values = [llm_process.get('memory_rss_mb', 0)]
    
    fig.add_trace(
        go.Bar(x=memory_labels, y=memory_values, name='Memory (MB)',
               marker=dict(color=['skyblue'])),
        row=1, col=1
    )
    
    # Process info table
    process_info = [
        ['Process ID', str(llm_process.get('pid', 'N/A'))],
        ['CPU Usage', f"{llm_process.get('cpu_percent', 0):.1f}%"],
        ['Memory %', f"{llm_process.get('memory_percent', 0):.1f}%"],
        ['RSS Memory', f"{llm_process.get('memory_rss_mb', 0):.1f} MB"]
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
    metrics_data = get_metrics_data()
    
    status_cards = create_status_cards(metrics_data)
    performance_charts = create_llm_performance_charts(metrics_data)
    process_charts = create_llm_process_charts(metrics_data)
    
    return status_cards, performance_charts, process_charts

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080) 