import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import pandas as pd
import requests
import threading
import time
import logging

from monitoring.config import get_config

logger = logging.getLogger(__name__)

# Global data storage
current_metrics = {}
metrics_history = []

config = get_config()
API_BASE_URL = f"http://{config.api.host}:{config.api.port}"

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "LLM Monitor"

# Simple layout
app.layout = html.Div([
    html.H1("LLM Performance Monitor"),
    
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    # Status and key metrics
    html.Div(id="status-info"),
    
    # Main chart
    dcc.Graph(id="main-chart"),
    
    # Basic metrics table
    html.Div(id="metrics-table")
])

def fetch_data():
    """Fetch essential data from API."""
    global current_metrics, metrics_history
    
    try:
        # Get current metrics
        response = requests.get(f"{API_BASE_URL}/metrics/current", timeout=5)
        if response.status_code == 200:
            current_metrics = response.json()
        
        # Get recent system metrics
        response = requests.get(f"{API_BASE_URL}/metrics/history?metric_type=system&hours=1", timeout=5)
        if response.status_code == 200:
            metrics_history = response.json()
            
    except Exception as e:
        logger.warning(f"API fetch failed: {e}")

def start_data_fetcher():
    """Start background data fetching."""
    def fetch_loop():
        while True:
            fetch_data()
            time.sleep(5)
    
    thread = threading.Thread(target=fetch_loop, daemon=True)
    thread.start()

@callback(Output('status-info', 'children'), [Input('interval-component', 'n_intervals')])
def update_status(n):
    """Update status and key metrics."""
    if not current_metrics:
        return html.P("No data available")
    
    system = current_metrics.get('system', {})
    performance = current_metrics.get('performance', {})
    
    return html.Div([
        html.H3("System Status"),
        html.P(f"CPU: {system.get('cpu_percent', 0):.1f}%"),
        html.P(f"Memory: {system.get('memory_percent', 0):.1f}%"), 
        html.P(f"Response Time: {performance.get('avg_response_time_ms', 0):.0f}ms"),
        html.P(f"Requests: {performance.get('total_requests', 0)}")
    ])

@callback(Output('main-chart', 'figure'), [Input('interval-component', 'n_intervals')])
def update_main_chart(n):
    """Update main performance chart."""
    if not metrics_history:
        return {'data': [], 'layout': {'title': 'No Data'}}
    
    df = pd.DataFrame(metrics_history)
    if df.empty:
        return {'data': [], 'layout': {'title': 'No Data'}}
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # CPU line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cpu_percent'],
        mode='lines',
        name='CPU %'
    ))
    
    # Memory line  
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['memory_percent'],
        mode='lines',
        name='Memory %'
    ))
    
    fig.update_layout(
        title='System Performance',
        xaxis_title='Time',
        yaxis_title='Percentage',
        height=400
    )
    
    return fig

@callback(Output('metrics-table', 'children'), [Input('interval-component', 'n_intervals')])
def update_metrics_table(n):
    """Update metrics table."""
    if not current_metrics:
        return html.P("No metrics data")
    
    system = current_metrics.get('system', {})
    performance = current_metrics.get('performance', {})
    
    return html.Table([
        html.Tr([html.Th("Metric"), html.Th("Value")]),
        html.Tr([html.Td("CPU Usage"), html.Td(f"{system.get('cpu_percent', 0):.1f}%")]),
        html.Tr([html.Td("Memory Usage"), html.Td(f"{system.get('memory_percent', 0):.1f}%")]),
        html.Tr([html.Td("Average Response Time"), html.Td(f"{performance.get('avg_response_time_ms', 0):.0f}ms")]),
        html.Tr([html.Td("Total Requests"), html.Td(f"{performance.get('total_requests', 0)}")]),
        html.Tr([html.Td("Success Rate"), html.Td(f"{performance.get('success_rate', 0):.1f}%")])
    ])

def run_dashboard():
    """Run the dashboard."""
    start_data_fetcher()
    app.run_server(
        host=config.dashboard.host,
        port=config.dashboard.port,
        debug=False
    )

if __name__ == "__main__":
    run_dashboard() 