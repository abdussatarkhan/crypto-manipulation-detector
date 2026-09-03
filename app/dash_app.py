"""
Real-Time Crypto Wash-Trade & Market Manipulation Surveillance Dashboard
========================================================================
Interactive Plotly Dash application featuring live trade streaming,
EWMA volume control charts, Benford's Law forensic bars, counterparty
network graph visualization, and real-time regulatory alert feeds.
"""

import os
import sys
import time
import datetime
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import load_config, setup_logger, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.graph_analysis import CryptoGraphAnalyzer
from scripts.anomaly_detection import StreamingAnomalyDetector
from scripts.benfords_law import BenfordsLawAnalyzer
from scripts.alert_scoring import CompositeAlertEngine

# Initialize Dash application
app = dash.Dash(
    __name__,
    title="Crypto Market Surveillance Engine",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

config = load_config()
logger = setup_logger("dash_app")

# Global streaming state buffer
TRADE_BUFFER = []
CURRENT_PRICE = 64500.0
ACTIVE_ALERTS = []
MAX_STREAM_HISTORY = 100


def seed_initial_state():
    """Seeds initial memory buffer with realistic trades and alerts."""
    global TRADE_BUFFER, ACTIVE_ALERTS
    raw_trades = generate_synthetic_trade_stream(num_trades=120, inject_wash_trading=True)
    prep = TradePreprocessor(config=config)
    clean_df = prep.normalize_trade_dataframe(raw_trades)
    TRADE_BUFFER = clean_df.to_dict(orient="records")

    engine = CompositeAlertEngine(config=config)
    report = engine.evaluate_dataset(clean_df)
    ACTIVE_ALERTS = report.get("alerts", [])


seed_initial_state()

# Application Layout
app.layout = html.Div([
    # Polling Timer for Live Updates (every 2 seconds)
    dcc.Interval(id="stream-interval", interval=2000, n_intervals=0),

    # Header Bar
    html.Div([
        html.Div([
            html.H1("CRYPTO MANIPULATION & WASH-TRADE RADAR", className="header-title"),
            html.Span("Real-Time Surveillance Engine", style={"color": "#8c9ba5", "fontSize": "0.85rem"})
        ]),
        html.Div([
            html.Span("● LIVE ENGINE RUNNING", className="live-badge"),
            html.Span(id="live-clock", style={"marginLeft": "15px", "fontFamily": "JetBrains Mono", "fontSize": "0.85rem", "color": "#8c9ba5"})
        ], style={"display": "flex", "alignItems": "center"})
    ], className="dash-header"),

    # Main Container
    html.Div([
        # Row 1: Key Performance Indicators (KPIs)
        html.Div([
            html.Div([
                html.Div("Manipulation Risk Index", className="kpi-label"),
                html.Div(id="kpi-mpi-val", className="kpi-value", style={"color": "#ff1744"}),
                html.Div(id="kpi-mpi-status", className="kpi-subtext")
            ], className="kpi-card"),

            html.Div([
                html.Div("Active Regulatory Alerts", className="kpi-label"),
                html.Div(id="kpi-alerts-val", className="kpi-value", style={"color": "#ff9100"}),
                html.Div("Critical & High Priority", className="kpi-subtext", style={"color": "#8c9ba5"})
            ], className="kpi-card"),

            html.Div([
                html.Div("Suspicious Wash Volume (Est.)", className="kpi-label"),
                html.Div(id="kpi-wash-vol-val", className="kpi-value", style={"color": "#00e5ff"}),
                html.Div("Detected in sliding window", className="kpi-subtext", style={"color": "#8c9ba5"})
            ], className="kpi-card"),

            html.Div([
                html.Div("Ingested Trades Analyzed", className="kpi-label"),
                html.Div(id="kpi-trades-count", className="kpi-value", style={"color": "#00e676"}),
                html.Div("Tick rate: ~45 trades/sec", className="kpi-subtext", style={"color": "#8c9ba5"})
            ], className="kpi-card")
        ], className="kpi-container"),

        # Row 2: Charts (EWMA Volume Surveillance + Benford's Law)
        html.Div([
            # EWMA Control Chart
            html.Div([
                html.Div([
                    html.Span("EWMA Control Chart: Volume & Dynamic Thresholds", className="card-title"),
                    html.Span("Control Limits (+3 Sigma)", style={"fontSize": "0.75rem", "color": "#8c9ba5"})
                ], style={"display": "flex", "justifyContent": "space-between"}),
                dcc.Graph(id="ewma-volume-graph", config={"displayModeBar": False}, style={"height": "320px"})
            ], className="dashboard-card", style={"flex": "2", "marginRight": "16px"}),

            # Benford's Law First-Digit Bar
            html.Div([
                html.Div([
                    html.Span("Benford's Law Forensic Conformity", className="card-title"),
                    html.Span("First-Digit Test (1-9)", style={"fontSize": "0.75rem", "color": "#8c9ba5"})
                ], style={"display": "flex", "justifyContent": "space-between"}),
                dcc.Graph(id="benford-bar-graph", config={"displayModeBar": False}, style={"height": "320px"})
            ], className="dashboard-card", style={"flex": "1"})
        ], style={"display": "flex", "flexWrap": "wrap"}),

        # Row 3: Counterparty Network Graph + Active Alerts Feed
        html.Div([
            # Counterparty Network
            html.Div([
                html.Div([
                    html.Span("Counterparty Transaction Graph (Wash Rings)", className="card-title"),
                    html.Span("Directed Flow & Cycles", style={"fontSize": "0.75rem", "color": "#8c9ba5"})
                ], style={"display": "flex", "justifyContent": "space-between"}),
                dcc.Graph(id="network-subgraph-view", config={"displayModeBar": False}, style={"height": "340px"})
            ], className="dashboard-card", style={"flex": "1.3", "marginRight": "16px"}),

            # Live Alerts Feed Table
            html.Div([
                html.Div([
                    html.Span("Real-Time Surveillance Alert Log", className="card-title"),
                    html.Span("Auto-Triage Active", style={"fontSize": "0.75rem", "color": "#ff1744"})
                ], style={"display": "flex", "justifyContent": "space-between"}),
                html.Div(id="live-alerts-table-container", style={"maxHeight": "300px", "overflowY": "auto"})
            ], className="dashboard-card", style={"flex": "1.7"})
        ], style={"display": "flex", "flexWrap": "wrap"})

    ], style={"padding": "24px 28px"})
])


@app.callback(
    [
        Output("live-clock", "children"),
        Output("kpi-mpi-val", "children"),
        Output("kpi-mpi-status", "children"),
        Output("kpi-alerts-val", "children"),
        Output("kpi-wash-vol-val", "children"),
        Output("kpi-trades-count", "children"),
        Output("ewma-volume-graph", "figure"),
        Output("benford-bar-graph", "figure"),
        Output("network-subgraph-view", "figure"),
        Output("live-alerts-table-container", "children")
    ],
    [Input("stream-interval", "n_intervals")]
)
def update_surveillance_dashboard(n_intervals):
    """Refreshes all charts, KPIs, and alerts on streaming tick intervals."""
    global TRADE_BUFFER, CURRENT_PRICE, ACTIVE_ALERTS

    # 1. Simulate arrival of new trade tick batch
    now_ms = int(time.time() * 1000)
    current_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")

    new_trades = generate_synthetic_trade_stream(num_trades=3, base_price=CURRENT_PRICE, inject_wash_trading=True)
    prep = TradePreprocessor(config=config)
    new_cleaned = prep.normalize_trade_dataframe(new_trades).to_dict(orient="records")

    TRADE_BUFFER.extend(new_cleaned)
    if len(TRADE_BUFFER) > MAX_STREAM_HISTORY:
        TRADE_BUFFER = TRADE_BUFFER[-MAX_STREAM_HISTORY:]

    df_current = pd.DataFrame(TRADE_BUFFER)
    CURRENT_PRICE = df_current["price"].iloc[-1]

    # 2. Run Anomaly Surveillance
    detector = StreamingAnomalyDetector(config=config)
    bars_df = prep.aggregate_to_bars(df_current, freq="1min")
    surveillance_df = detector.run_full_pipeline(bars_df)

    # 3. Benford's Law
    benford_analyzer = BenfordsLawAnalyzer(config=config)
    benford_res = benford_analyzer.analyze_distribution(df_current["quantity"])

    # 4. Graph & Wash Ring Detection
    graph_analyzer = CryptoGraphAnalyzer(config=config)
    graph = graph_analyzer.build_transaction_graph(df_current)
    cycles = graph_analyzer.detect_wash_trading_cycles(max_length=4)

    # 5. Composite Scoring
    engine = CompositeAlertEngine(config=config)
    audit = engine.evaluate_dataset(df_current)
    mpi = audit["manipulation_probability_index"]
    severity = audit["overall_severity"]

    # Calculate estimated wash volume
    wash_vol_usd = sum([c["total_cycle_volume_usd"] for c in cycles])
    wash_vol_usd += df_current[df_current["buyer_address"].str.contains("WASH", na=False)]["notional_usd"].sum()

    # KPI Outputs
    kpi_mpi = f"{mpi * 100:.1f}%"
    kpi_status = html.Span(f"Status: {severity}", className=f"badge-{severity.lower()}")
    kpi_alerts = f"{audit['active_alerts_count']}"
    kpi_wash = f"${wash_vol_usd:,.0f}"
    kpi_count = f"{len(df_current):,}"

    # 6. EWMA Figure
    fig_ewma = go.Figure()
    if not surveillance_df.empty:
        fig_ewma.add_trace(go.Scatter(
            x=surveillance_df["datetime_utc"],
            y=surveillance_df["volume"],
            mode="lines+markers",
            name="Observed Volume",
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=4)
        ))
        fig_ewma.add_trace(go.Scatter(
            x=surveillance_df["datetime_utc"],
            y=surveillance_df["ewma_volume_ucl"],
            mode="lines",
            name="EWMA UCL (+3σ)",
            line=dict(color="#ff1744", width=2, dash="dot")
        ))
        fig_ewma.add_trace(go.Scatter(
            x=surveillance_df["datetime_utc"],
            y=surveillance_df["ewma_volume_mean"],
            mode="lines",
            name="EWMA Mean",
            line=dict(color="#ffd600", width=1.5, dash="dash")
        ))

    fig_ewma.update_layout(
        paper_bgcolor="#121826",
        plot_bgcolor="#121826",
        font=dict(color="#8c9ba5", family="Inter"),
        margin=dict(l=40, r=20, t=20, b=30),
        xaxis=dict(gridcolor="#1a2234", zeroline=False),
        yaxis=dict(gridcolor="#1a2234", zeroline=False, title="Volume (BTC)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 7. Benford Figure
    digits = list(range(1, 10))
    exp_pcts = [benford_analyzer.BENFORD_PROBABILITIES[d] * 100 for d in digits]
    obs_pcts = [benford_res["empirical_proportions"].get(d, 0.0) * 100 for d in digits] if "empirical_proportions" in benford_res else [0]*9

    fig_benford = go.Figure()
    fig_benford.add_trace(go.Bar(
        x=digits,
        y=obs_pcts,
        name="Empirical Trade Sizes",
        marker_color="#ff1744" if benford_res.get("ks_rejected") else "#00e676"
    ))
    fig_benford.add_trace(go.Scatter(
        x=digits,
        y=exp_pcts,
        mode="lines+markers",
        name="Benford's Law Curve",
        line=dict(color="#00e5ff", width=2.5),
        marker=dict(size=6)
    ))
    fig_benford.update_layout(
        paper_bgcolor="#121826",
        plot_bgcolor="#121826",
        font=dict(color="#8c9ba5", family="Inter"),
        margin=dict(l=40, r=20, t=20, b=30),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1, gridcolor="#1a2234", title="Leading Digit"),
        yaxis=dict(gridcolor="#1a2234", title="Frequency (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 8. Counterparty Graph Figure
    fig_network = go.Figure()
    if graph.number_of_nodes() > 0:
        pos = nx.spring_layout(graph, seed=42, k=0.5)

        # Edges
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig_network.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color="#242f48"),
            hoverinfo="none",
            mode="lines"
        ))

        # Nodes
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_color = ["#ff1744" if "WASH" in str(node) else "#00e5ff" for node in graph.nodes()]

        fig_network.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[str(node)[:6] for node in graph.nodes()],
            textposition="top center",
            marker=dict(
                color=node_color,
                size=12,
                line=dict(width=1.5, color="#ffffff")
            )
        ))

    fig_network.update_layout(
        paper_bgcolor="#121826",
        plot_bgcolor="#121826",
        font=dict(color="#8c9ba5", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )

    # 9. Alerts Table Component
    alerts_rows = []
    for a in audit.get("alerts", [])[:8]:
        sev_class = f"badge-{a['severity'].lower()}"
        alerts_rows.append(html.Tr([
            html.Td(a["alert_id"], className="font-mono"),
            html.Td(html.Span(a["severity"], className=sev_class)),
            html.Td(a["type"]),
            html.Td(a["description"], style={"fontSize": "0.78rem"}),
            html.Td(f"${a.get('volume_usd', 0):,.0f}", className="font-mono")
        ]))

    alerts_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Alert ID"),
            html.Th("Severity"),
            html.Th("Type"),
            html.Th("Details"),
            html.Th("Notional")
        ])),
        html.Tbody(alerts_rows)
    ], className="alert-table")

    return (
        current_time_str,
        kpi_mpi,
        kpi_status,
        kpi_alerts,
        kpi_wash,
        kpi_count,
        fig_ewma,
        fig_benford,
        fig_network,
        alerts_table
    )


if __name__ == "__main__":
    host = config.get("dashboard", {}).get("host", "127.0.0.1")
    port = config.get("dashboard", {}).get("port", 8050)
    print(f"Starting Crypto Manipulation Surveillance Dashboard at http://{host}:{port} ...")
    app.run_server(host=host, port=port, debug=False)
