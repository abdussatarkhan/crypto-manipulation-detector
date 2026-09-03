# %% [markdown]
# # Notebook 02: Counterparty Graph Analysis & Wash-Trading Ring Detection
# 
# **Project**: Real-Time Crypto Wash-Trade & Market Manipulation Detector  
# **Objective**: Model crypto counterparty transactions as a directed weighted network,
# apply Louvain community detection to identify colluding syndicates, execute cycle detection
# algorithms to expose circular wash-trading loops, and calculate network centrality.

# %%
import os
import sys
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import load_config, setup_logger, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.graph_analysis import CryptoGraphAnalyzer

config = load_config()
logger = setup_logger("nb_02_graph")
print("Initialized Graph Analysis environment.")

# %% [markdown]
# ## 1. Ingest Data & Construct Directed Counterparty Graph
# 
# Every node in the graph represents a wallet address or exchange deposit point.
# Every directed edge $(u \to v)$ represents cumulative trade volume transferred from seller $u$ to buyer $v$.

# %%
stream = generate_synthetic_trade_stream(num_trades=800, inject_wash_trading=True, random_seed=42)
preprocessor = TradePreprocessor(config=config)
trades_df = preprocessor.normalize_trade_dataframe(stream)

analyzer = CryptoGraphAnalyzer(config=config)
graph = analyzer.build_transaction_graph(trades_df)

print(f"Network Nodes (Wallets): {graph.number_of_nodes()}")
print(f"Network Edges (Flows):   {graph.number_of_edges()}")
print(f"Network Density:         {nx.density(graph):.4f}")
print(f"Reciprocity:             {nx.reciprocity(graph):.4f}")

# %% [markdown]
# ## 2. Wash-Trading Cycle Detection
# 
# A primary signature of wash-trading syndicates is circular fund/token movement:
# Address $A \to B \to C \to A$, where tokens circulate to artificially inflate trade volume
# without shifting economic risk.

# %%
detected_cycles = analyzer.detect_wash_trading_cycles(max_length=5)
print(f"\nDiscovered {len(detected_cycles)} potential wash-trading cycles:")
for c in detected_cycles[:10]:
    print(f"[{c['cycle_id']}] Length: {c['length']} | Risk Score: {c['risk_score']:.2f} | Volume: ${c['total_cycle_volume_usd']:,.2f}")
    print(f"       Path: {c['cycle_path']}")

# %% [markdown]
# ## 3. Louvain Community Detection (Syndicate Partitioning)
# 
# We partition the network into modular communities using the Louvain heuristic.
# Colluding wash rings form densely intra-connected clusters with high internal modularity.

# %%
communities = analyzer.run_community_detection()
num_communities = len(set(communities.values()))
print(f"\nPartitioned into {num_communities} distinct trading communities.")

# Count members per community
comm_counts = pd.Series(communities).value_counts()
print("\nTop 5 Communities by Member Count:")
print(comm_counts.head())

# %% [markdown]
# ## 4. Centrality Analysis & Ringleader Identification
# 
# Calculate Betweenness Centrality, In-Degree, Out-Degree, and Syndicate Orchestrator scores.
# Syndicate hubs act as brokers or liquidity funnels between ring participants.

# %%
centrality_df = analyzer.compute_centrality_metrics()
print("\n--- TOP 10 SUSPECTED SYNDICATE ORCHESTRATORS ---")
print(centrality_df.head(10)[["address", "community_id", "total_volume_usd", "betweenness_centrality", "syndicate_orchestrator_score"]])

# %% [markdown]
# ## 5. Graph Visualization & Syndicate Subgraph Isolation
# 
# Extract and visualize the highest-risk syndicate cluster, coloring nodes by community
# and highlighting wash-trading cycle edges in red.

# %%
plt.figure(figsize=(12, 10))

# Select a compact subgraph of active nodes
top_nodes = list(centrality_df.head(25)["address"])
subgraph = graph.subgraph(top_nodes).copy()

pos = nx.spring_layout(subgraph, k=0.6, seed=42)

# Extract node communities
node_colors = [communities.get(n, 0) for n in subgraph.nodes()]
node_sizes = [subgraph.nodes[n].get("in_volume_usd", 100) / 1000 + 300 for n in subgraph.nodes()]

# Draw base network
nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.tab10, alpha=0.9)
nx.draw_networkx_labels(subgraph, pos, labels={n: n[:8] for n in subgraph.nodes()}, font_size=8, font_color="black")
nx.draw_networkx_edges(subgraph, pos, edge_color="gray", arrows=True, arrowsize=12, alpha=0.5, width=1.2)

# Highlight detected wash-trading cycle edges in crimson
wash_nodes_set = set()
for c in detected_cycles[:3]:
    wash_nodes_set.update(c["nodes"])
    cycle_edges = [(c["nodes"][i], c["nodes"][(i+1)%len(c["nodes"])]) for i in range(len(c["nodes"]))]
    valid_edges = [e for e in cycle_edges if subgraph.has_edge(e[0], e[1])]
    nx.draw_networkx_edges(subgraph, pos, edgelist=valid_edges, edge_color="crimson", width=3.0, arrows=True, arrowsize=16)

plt.title("Cryptocurrency Counterparty Graph: Suspicious Wash-Trading Rings Highlighted", fontsize=14, fontweight="bold")
plt.axis("off")
plt.show()

print("Graph analysis walkthrough completed successfully.")
