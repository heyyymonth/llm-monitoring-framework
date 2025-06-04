import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import requests
import threading
import time
import logging

from monitoring.config import get_config

logger = logging.getLogger(__name__)

# Global data storage
current_metrics = {}

config = get_config()
API_BASE_URL = f"http://{config.api.host}:{config.api.port}"

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "LLM Monitor"

# Minimal layout
app.layout = html.Div([
    html.H1("LLM Performance Monitor", style={'text-align': 'center', 'margin-bottom': '30px'}),
    
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    # Status cards
    html.Div(id="status-cards", style={'margin-bottom': '20px'}),
    
    # Performance chart
    dcc.Graph(id="performance-chart"),
    
    # Metrics table
    html.Div(id="metrics-table")
], style={'padding': '20px', 'max-width': '1200px', 'margin': '0 auto'})

def fetch_data():
    """Fetch data from API."""
    global current_metrics
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics/current", timeout=5)
        if response.status_code == 200:
            current_metrics = response.json()
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

# Start data fetcher
start_data_fetcher()

@callback(Output('status-cards', 'children'), [Input('interval-component', 'n_intervals')])
def update_status_cards(n):
    """Update status cards."""
    if not current_metrics:
        return html.Div("Loading...", style={'text-align': 'center', 'color': '#666'})
    
    system = current_metrics.get('system', {})
    performance = current_metrics.get('performance', {})
    status = current_metrics.get('status', 'unknown')
    
    # Status color
    status_color = '#28a745' if status == 'healthy' else '#ffc107' if status == 'degraded' else '#dc3545'
    
    cards = [
        html.Div([
            html.H4(f"{status.upper()}", style={'color': status_color, 'margin': '0'}),
            html.P("System Status", style={'margin': '5px 0 0 0', 'color': '#666'})
        ], style={'text-align': 'center', 'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
        
        html.Div([
            html.H4(f"{system.get('cpu_percent', 0):.1f}%", style={'margin': '0', 'color': '#007bff'}),
            html.P("CPU Usage", style={'margin': '5px 0 0 0', 'color': '#666'})
        ], style={'text-align': 'center', 'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
        
        html.Div([
            html.H4(f"{system.get('memory_percent', 0):.1f}%", style={'margin': '0', 'color': '#17a2b8'}),
            html.P("Memory Usage", style={'margin': '5px 0 0 0', 'color': '#666'})
        ], style={'text-align': 'center', 'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
        
        html.Div([
            html.H4(f"{performance.get('avg_response_time_ms', 0):.0f}ms", style={'margin': '0', 'color': '#28a745'}),
            html.P("Avg Response Time", style={'margin': '5px 0 0 0', 'color': '#666'})
        ], style={'text-align': 'center', 'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px'}),
        
        html.Div([
            html.H4(f"{performance.get('total_requests', 0)}", style={'margin': '0', 'color': '#6f42c1'}),
            html.P("Total Requests", style={'margin': '5px 0 0 0', 'color': '#666'})
        ], style={'text-align': 'center', 'padding': '15px', 'border': '1px solid #ddd', 'border-radius': '5px'})
    ]
    
    return html.Div(cards, style={
        'display': 'grid', 
        'grid-template-columns': 'repeat(auto-fit, minmax(200px, 1fr))', 
        'gap': '15px'
    })

@callback(Output('performance-chart', 'figure'), [Input('interval-component', 'n_intervals')])
def update_chart(n):
    """Update performance chart."""
    if not current_metrics:
        return {'data': [], 'layout': {'title': 'Loading...', 'height': 400}}
    
    performance = current_metrics.get('performance', {})
    system = current_metrics.get('system', {})
    
    # Simple bar chart
    metrics = [
        ('Requests', performance.get('total_requests', 0)),
        ('Avg Response (ms)', performance.get('avg_response_time_ms', 0) / 10),  # Scale for visibility
        ('CPU %', system.get('cpu_percent', 0)),
        ('Memory %', system.get('memory_percent', 0)),
        ('Tokens/sec', performance.get('avg_tokens_per_second', 0))
    ]
    
    fig = go.Figure([go.Bar(
        x=[m[0] for m in metrics],
        y=[m[1] for m in metrics],
        marker_color=['#6f42c1', '#28a745', '#007bff', '#17a2b8', '#ffc107']
    )])
    
    fig.update_layout(
        title='Performance Metrics',
        height=400,
        showlegend=False,
        margin={'l': 40, 'r': 40, 't': 60, 'b': 40}
    )
    
    return fig

@callback(Output('metrics-table', 'children'), [Input('interval-component', 'n_intervals')])
def update_table(n):
    """Update metrics table."""
    if not current_metrics:
        return html.Div("Loading metrics...", style={'text-align': 'center', 'color': '#666'})
    
    system = current_metrics.get('system', {})
    performance = current_metrics.get('performance', {})
    
    # Calculate success rate
    total_req = performance.get('total_requests', 0)
    success_req = performance.get('successful_requests', 0)
    success_rate = (success_req / total_req * 100) if total_req > 0 else 0
    
    table_data = [
        ("CPU Usage", f"{system.get('cpu_percent', 0):.1f}%"),
        ("Memory Usage", f"{system.get('memory_percent', 0):.1f}%"),
        ("Total Requests", f"{performance.get('total_requests', 0)}"),
        ("Success Rate", f"{success_rate:.1f}%"),
        ("Avg Response Time", f"{performance.get('avg_response_time_ms', 0):.0f}ms"),
        ("Tokens/sec", f"{performance.get('avg_tokens_per_second', 0):.1f}"),
        ("Total Tokens", f"{performance.get('total_tokens_processed', 0)}")
    ]
    
    rows = [html.Tr([html.Th("Metric"), html.Th("Value")], style={'background-color': '#f8f9fa'})]
    for metric, value in table_data:
        rows.append(html.Tr([html.Td(metric), html.Td(value)]))
    
    return html.Div([
        html.H3("Detailed Metrics", style={'margin-bottom': '15px'}),
        html.Table(rows, style={
            'width': '100%',
            'border-collapse': 'collapse',
            'border': '1px solid #ddd'
        })
    ])

def run_dashboard():
    """Run the dashboard."""
    app.run_server(
        host=config.dashboard.host,
        port=config.dashboard.port,
        debug=False
    )

if __name__ == "__main__":
    run_dashboard() 