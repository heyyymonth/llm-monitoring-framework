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
API_BASE_URL = "http://llm-monitor-api:8000"
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
        # Fetch real LLM monitoring metrics
        quality_response = requests.get(f"{API_BASE_URL}/metrics/quality", timeout=5)
        safety_response = requests.get(f"{API_BASE_URL}/metrics/safety", timeout=5)
        cost_response = requests.get(f"{API_BASE_URL}/metrics/cost", timeout=5)
        
        if all(r.status_code == 200 for r in [quality_response, safety_response, cost_response]):
            data = {
                "quality": quality_response.json(),
                "safety": safety_response.json(), 
                "cost": cost_response.json(),
                "status": "healthy"
            }
            return data
    except Exception as e:
        logger.error(f"Error fetching metrics data: {e}")
    return None

def get_current_metrics():
    """Get current metrics for display."""
    data = get_metrics_data()
    
    if data and data.get("status") == "healthy":
        quality_data = data.get("quality", {})
        safety_data = data.get("safety", {})
        cost_data = data.get("cost", {})
        
        # Calculate safety score as 1 - violation_rate
        violation_rate = safety_data.get("violation_rate", 1.0)
        safety_score = 1.0 - violation_rate

        metrics = {
            "quality_score": quality_data.get("average_quality", 0.0),
            "safety_score": safety_score,
            "cost_usd": cost_data.get("total_cost_usd", 0.0),
            "total_requests": safety_data.get("total_interactions", 0),
            "active_flags": len(safety_data.get("common_flags", [])),
            "cost_today": cost_data.get("cost_by_period", {}).get("today", 0.0),
            "common_flags": safety_data.get("common_flags", []),
            "total_violations": safety_data.get("safety_violations", 0),
            "cost_by_model": cost_data.get("cost_by_model", {})
        }
        return metrics
    
    return {
        "quality_score": 0.0,
        "safety_score": 0.0, 
        "cost_usd": 0.0,
        "total_requests": 0,
        "active_flags": 0,
        "cost_today": 0.0,
        "common_flags": [],
        "total_violations": 0,
        "cost_by_model": {}
    }

def create_status_cards(metrics_data):
    """Create status cards for key LLM metrics."""
    if not metrics_data:
        return html.Div("No data available", className="alert alert-danger")
    
    current_metrics = get_current_metrics()
    
    # Calculate status colors
    def get_color(value, good_threshold, warning_threshold):
        if value >= good_threshold:
            return "#28a745"  # Green
        elif value >= warning_threshold:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    
    quality_color = get_color(current_metrics['quality_score'], 0.7, 0.5)
    safety_color = get_color(current_metrics['safety_score'], 0.8, 0.6)
    
    cards = [
        # Quality Score
        html.Div([
            html.H5("🎯 Quality Score", className="card-title"),
            html.H3(f"{current_metrics['quality_score']:.3f}", 
                    style={'color': quality_color}),
            html.P("Response quality assessment")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Safety Score
        html.Div([
            html.H5("🛡️ Safety Score", className="card-title"),
            html.H3(f"{current_metrics['safety_score']:.3f}", 
                    style={'color': safety_color}),
            html.P(f"Active flags: {current_metrics['active_flags']}")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Total Cost
        html.Div([
            html.H5("💰 Total Cost", className="card-title"),
            html.H3(f"${current_metrics['cost_usd']:.6f}"),
            html.P("Cumulative inference cost")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Total Requests
        html.Div([
            html.H5("📊 Total Requests", className="card-title"),
            html.H3(f"{current_metrics['total_requests']}"),
            html.P("LLM inference requests")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Cost Today
        html.Div([
            html.H5("💸 Cost Today", className="card-title"),
            html.H3(f"${current_metrics['cost_today']:.6f}", 
                    style={'color': '#28a745' if current_metrics['cost_today'] < 1.0 else '#ffc107'}),
            html.P("Today's spending")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
        
        # Status Indicator
        html.Div([
            html.H5("🔄 Status", className="card-title"),
            html.H3("✅ Active" if metrics_data else "❌ Offline"),
            html.P("Monitoring service")
        ], className="card text-center", style={'padding': '1rem', 'margin': '0.5rem'}),
    ]
    
    return html.Div(cards, className="row")

def create_llm_performance_charts(metrics_data):
    """Create LLM quality and safety charts."""
    if not metrics_data:
        return html.Div("No LLM data available", className="alert alert-warning")
    
    current_metrics = get_current_metrics()
    quality_data = metrics_data.get('quality', {})
    safety_data = metrics_data.get('safety', {})
    cost_data = metrics_data.get('cost', {})
    
    # Create subplots for LLM-specific metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Quality & Safety Scores', 'Cost Breakdown', 'Safety Flags Over Time', 'Quality Distribution'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Quality & Safety Scores
    fig.add_trace(
        go.Bar(x=['Quality Score', 'Safety Score'], 
               y=[current_metrics['quality_score'], current_metrics['safety_score']],
               name='LLM Performance',
               marker=dict(color=['skyblue', 'lightgreen'])),
        row=1, col=1
    )
    
    # Cost Breakdown
    cost_by_model = current_metrics.get('cost_by_model', {})
    if cost_by_model:
        fig.add_trace(
            go.Pie(labels=list(cost_by_model.keys()), 
                values=list(cost_by_model.values()),
                name="Cost by Model"),
            row=1, col=2
        )
    
    # Safety Flags (simplified)
    common_flags = current_metrics.get("common_flags", [])
    total_violations = current_metrics.get("total_violations", 0)
    total_requests = current_metrics.get("total_requests", 0)

    flag_counts = {flag: 0 for flag in common_flags} # Placeholder
    flag_counts['No Issues'] = max(0, total_requests - total_violations)
    
    if total_violations > 0 and common_flags:
        # For simplicity, we can distribute total violations among common flags
        # This is not accurate but serves for visualization
        per_flag_count = total_violations // len(common_flags)
        for flag in common_flags:
            flag_counts[flag] = per_flag_count

    fig.add_trace(
        go.Bar(x=list(flag_counts.keys()), 
               y=list(flag_counts.values()),
               name='Safety Issues',
               marker=dict(color='lightcoral')),
        row=2, col=1
    )
    
    # Quality Distribution (mock trending data)
    quality_trend = quality_data.get('quality_trend', [current_metrics['quality_score']] * 5)
    fig.add_trace(
        go.Bar(x=[f'T-{i}' for i in range(4, -1, -1)], 
               y=quality_trend,
               name='Quality Trend',
               marker=dict(color='gold')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="LLM Quality & Safety Monitoring")
    return dcc.Graph(figure=fig)

def create_llm_process_charts(metrics_data):
    """Create LLM cost and request statistics."""
    if not metrics_data:
        return html.Div("No cost data available", className="alert alert-warning")
    
    current_metrics = get_current_metrics()
    cost_data = metrics_data.get('cost', {})
    
    # Create cost analytics display
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cost Summary', 'Request Statistics'),
        specs=[[{"type": "bar"}, {"type": "table"}]]
    )
    
    # Cost breakdown by time period
    cost_periods = ['Today', 'Total', 'Projected Monthly']
    cost_values = [
        current_metrics['cost_today'],
        current_metrics['cost_usd'], 
        current_metrics['cost_usd'] * 30  # Simple projection
    ]
    
    fig.add_trace(
        go.Bar(x=cost_periods, y=cost_values, name='Cost ($)',
               marker=dict(color=['lightblue', 'darkblue', 'orange'])),
        row=1, col=1
    )
    
    # LLM statistics table
    llm_stats = [
        ['Total Requests', str(current_metrics['total_requests'])],
        ['Quality Score', f"{current_metrics['quality_score']:.3f}"],
        ['Safety Score', f"{current_metrics['safety_score']:.3f}"],
        ['Active Flags', str(current_metrics['active_flags'])],
        ['Total Cost', f"${current_metrics['cost_usd']:.6f}"],
        ['Cost Today', f"${current_metrics['cost_today']:.6f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='lightgray',
                       align='left'),
            cells=dict(values=list(zip(*llm_stats)),
                      fill_color='white',
                      align='left')),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="LLM Cost & Request Analytics")
    return dcc.Graph(figure=fig)

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🧠 LLM Quality & Safety Monitor", className="text-center mb-4"),
        html.P("Real-time monitoring of LLM quality, safety, and cost metrics", 
               className="text-center text-muted")
    ], className="container mt-3"),
    
    # Status Cards
    html.Div([
        html.H3("📊 Current Status", className="mb-3"),
        html.Div(id="status-cards")
    ], className="container mb-4"),
    
    # Main Charts
    html.Div([
        html.H3("📈 Quality & Safety Analytics", className="mb-3"),
        html.Div(id="llm-performance-charts")
    ], className="container mb-4"),
    
    # Process Charts
    html.Div([
        html.H3("💰 Cost & Request Analytics", className="mb-3"),
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
    app.run(debug=False, host='0.0.0.0', port=8080) 