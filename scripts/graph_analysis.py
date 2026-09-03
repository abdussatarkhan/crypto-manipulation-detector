"""
On-Chain & Counterparty Graph Analytics for Wash-Trade Ring Detection
=====================================================================
Builds directed multi-edge counterparty transaction graphs using NetworkX,
identifies cyclical wash-trading loops (A -> B -> C -> A), extracts tightly knit
syndicate clusters using Louvain community detection, and calculates graph centrality.
"""

import os
import sys
import argparse
import itertools
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional

# Optional community detection library
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import setup_logger, load_config, get_project_root, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor


class CryptoGraphAnalyzer:
    """Network analysis engine for cryptocurrency counterparty flow graphs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        self.config = config or load_config()
        self.logger = logger or setup_logger("graph_analyzer")
        self.graph_cfg = self.config.get("graph_analysis", {})
        self.max_cycle_length = self.graph_cfg.get("max_cycle_length", 5)
        self.min_cycle_usd = self.graph_cfg.get("min_cycle_volume_usd", 1000.0)

        self.graph = nx.DiGraph()

    def build_transaction_graph(self, trades_df: pd.DataFrame) -> nx.DiGraph:
        """
        Builds a directed weighted graph where nodes represent wallet addresses
        and edges represent cumulative volume sent from seller to buyer (or on-chain sender to recipient).

        Args:
            trades_df: Preprocessed DataFrame containing buyer_address, seller_address, notional_usd.

        Returns:
            Directed NetworkX DiGraph.
        """
        self.graph.clear()
        self.logger.info(f"Building counterparty transaction graph from {len(trades_df)} trades...")

        # Aggregate pairwise volumes
        grouped = trades_df.groupby(["seller_address", "buyer_address"]).agg(
            total_volume_usd=("notional_usd", "sum"),
            tx_count=("trade_id", "count"),
            avg_price=("price", "mean")
        ).reset_index()

        for _, row in grouped.iterrows():
            u = str(row["seller_address"])
            v = str(row["buyer_address"])
            vol = float(row["total_volume_usd"])
            count = int(row["tx_count"])

            if u == v:
                # Self-loop
                self.graph.add_edge(u, v, weight=vol, tx_count=count, is_self_loop=True)
            else:
                self.graph.add_edge(u, v, weight=vol, tx_count=count, is_self_loop=False)

        # Annotate node metadata
        for node in self.graph.nodes:
            in_deg = self.graph.in_degree(node, weight="weight")
            out_deg = self.graph.out_degree(node, weight="weight")
            self.graph.nodes[node]["in_volume_usd"] = float(in_deg)
            self.graph.nodes[node]["out_volume_usd"] = float(out_deg)
            self.graph.nodes[node]["net_flow_usd"] = float(in_deg - out_deg)

        self.logger.info(f"Constructed DiGraph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def detect_wash_trading_cycles(self, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Detects directed circular trade paths (cycles) indicative of wash trading.
        e.g., A -> B -> C -> A where funds circulate with minimal economic purpose.

        Args:
            max_length: Maximum cycle depth (default from config, e.g. 5).

        Returns:
            List of detected wash cycles with participating nodes and volume metrics.
        """
        limit = max_length or self.max_cycle_length
        detected_cycles = []

        self.logger.info(f"Detecting wash trading cycles up to length {limit}...")

        # Find simple cycles using Johnson's algorithm
        try:
            raw_cycles = list(nx.simple_cycles(self.graph))
        except Exception as e:
            self.logger.error(f"Cycle detection error: {e}")
            return []

        for cycle in raw_cycles:
            cycle_len = len(cycle)
            if 1 <= cycle_len <= limit:
                # Calculate cycle bottleneck volume and edge weights
                cycle_edges = []
                edge_weights = []
                for i in range(cycle_len):
                    u = cycle[i]
                    v = cycle[(i + 1) % cycle_len]
                    if self.graph.has_edge(u, v):
                        w = self.graph[u][v].get("weight", 0.0)
                        tx_c = self.graph[u][v].get("tx_count", 0)
                        edge_weights.append(w)
                        cycle_edges.append({"from": u, "to": v, "volume_usd": w, "tx_count": tx_c})

                min_flow = min(edge_weights) if edge_weights else 0.0
                total_circulated = sum(edge_weights)

                # Flag as wash if bottleneck volume exceeds minimum threshold
                if total_circulated >= self.min_cycle_usd or cycle_len == 1:
                    risk_score = 1.0 if cycle_len == 1 else min(0.95, 0.60 + 0.08 * (cycle_len <= 3) + 0.15 * (min_flow > 5000))
                    detected_cycles.append({
                        "cycle_id": f"CYC-{len(detected_cycles) + 1}",
                        "length": cycle_len,
                        "nodes": cycle,
                        "cycle_path": " -> ".join(cycle + [cycle[0]]),
                        "bottleneck_volume_usd": round(min_flow, 2),
                        "total_cycle_volume_usd": round(total_circulated, 2),
                        "risk_score": round(risk_score, 3),
                        "is_self_trade": (cycle_len == 1)
                    })

        self.logger.info(f"Discovered {len(detected_cycles)} potential wash-trading cycles.")
        return detected_cycles

    def run_community_detection(self) -> Dict[str, int]:
        """
        Executes Louvain modularity-based community detection on an undirected projection
        of the transaction graph to partition nodes into trading syndicates.

        Returns:
            Dictionary mapping wallet address -> community ID.
        """
        # Create undirected weighted graph for modularity optimization
        undirected_g = self.graph.to_undirected()
        for u, v, data in undirected_g.edges(data=True):
            # Sum forward and backward weights
            fwd_w = self.graph[u][v]["weight"] if self.graph.has_edge(u, v) else 0.0
            rev_w = self.graph[v][u]["weight"] if self.graph.has_edge(v, u) else 0.0
            data["weight"] = fwd_w + rev_w

        if len(undirected_g.nodes) == 0:
            return {}

        if LOUVAIN_AVAILABLE:
            self.logger.info("Running Louvain community detection...")
            partition = community_louvain.best_partition(
                undirected_g,
                weight="weight",
                resolution=self.graph_cfg.get("louvain", {}).get("resolution", 1.0),
                random_state=42
            )
        else:
            self.logger.info("Using NetworkX greedy modularity communities...")
            communities_gen = nx.community.greedy_modularity_communities(undirected_g, weight="weight")
            partition = {}
            for comm_id, comm_nodes in enumerate(communities_gen):
                for node in comm_nodes:
                    partition[node] = comm_id

        # Attach community attribute to nodes
        for node, comm_id in partition.items():
            if self.graph.has_node(node):
                self.graph.nodes[node]["community_id"] = comm_id

        num_communities = len(set(partition.values()))
        self.logger.info(f"Identified {num_communities} distinct trading communities.")
        return partition

    def compute_centrality_metrics(self) -> pd.DataFrame:
        """
        Calculates network centrality metrics including Betweenness, In-Degree,
        Out-Degree, and PageRank to spot orchestrators of wash networks.

        Returns:
            pd.DataFrame ranking wallets by suspicious network prominence.
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame()

        self.logger.info("Calculating network centrality metrics...")

        # Centrality algorithms
        in_degree = dict(self.graph.in_degree(weight="weight"))
        out_degree = dict(self.graph.out_degree(weight="weight"))
        betweenness = nx.betweenness_centrality(self.graph, weight="weight", normalized=True)
        pagerank = nx.pagerank(self.graph, weight="weight", alpha=0.85)
        clustering = nx.clustering(self.graph.to_undirected())

        records = []
        for node in self.graph.nodes:
            in_w = in_degree.get(node, 0.0)
            out_w = out_degree.get(node, 0.0)
            tot_w = in_w + out_w
            # High flow with low net imbalance is characteristic of wash orchestrators
            imbalance = abs(in_w - out_w) / (tot_w + 1e-6)
            syndicate_score = (betweenness.get(node, 0.0) * 0.4 +
                               pagerank.get(node, 0.0) * 0.3 +
                               (1.0 - imbalance) * 0.3)

            records.append({
                "address": node,
                "community_id": self.graph.nodes[node].get("community_id", -1),
                "in_volume_usd": round(in_w, 2),
                "out_volume_usd": round(out_w, 2),
                "total_volume_usd": round(tot_w, 2),
                "flow_imbalance": round(imbalance, 3),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 5),
                "pagerank": round(pagerank.get(node, 0.0), 5),
                "clustering_coef": round(clustering.get(node, 0.0), 4),
                "syndicate_orchestrator_score": round(syndicate_score, 4)
            })

        df_metrics = pd.DataFrame(records)
        df_metrics.sort_values("syndicate_orchestrator_score", ascending=False, inplace=True)
        df_metrics.reset_index(drop=True, inplace=True)
        return df_metrics

    def export_graphml(self, filepath: str):
        """Exports the graph to GraphML format for Gephi or Cytoscape visualization."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Ensure all node/edge attributes are strings, floats, ints, or bools
        clean_g = nx.DiGraph()
        for u, v, d in self.graph.edges(data=True):
            clean_g.add_edge(str(u), str(v), **{k: (str(val) if not isinstance(val, (int, float, bool, str)) else val) for k, val in d.items()})
        for n, d in self.graph.nodes(data=True):
            clean_g.add_node(str(n), **{k: (str(val) if not isinstance(val, (int, float, bool, str)) else val) for k, val in d.items()})

        nx.write_graphml(clean_g, filepath)
        self.logger.info(f"Exported GraphML structure to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Analyze crypto transaction graph for wash cycles & syndicates.")
    parser.add_argument("--trades", type=str, default=None, help="Path to cleaned trades parquet/csv")
    parser.add_argument("--export-gephi", action="store_true", help="Export GraphML for Gephi")
    args = parser.parse_args()

    root = get_project_root()
    analyzer = CryptoGraphAnalyzer()

    if args.trades and os.path.exists(args.trades):
        trades_df = pd.read_parquet(args.trades) if args.trades.endswith(".parquet") else pd.read_csv(args.trades)
    else:
        analyzer.logger.info("Using simulated trade data with injected wash-trading rings.")
        synthetic_data = generate_synthetic_trade_stream(num_trades=600, inject_wash_trading=True)
        preprocessor = TradePreprocessor()
        trades_df = preprocessor.normalize_trade_dataframe(synthetic_data)

    analyzer.build_transaction_graph(trades_df)
    cycles = analyzer.detect_wash_trading_cycles()
    communities = analyzer.run_community_detection()
    centrality_df = analyzer.compute_centrality_metrics()

    print("\n--- TOP SUSPECTED SYNDICATE NODES ---")
    print(centrality_df.head(10)[["address", "community_id", "total_volume_usd", "flow_imbalance", "syndicate_orchestrator_score"]])

    print(f"\n--- DETECTED WASH CYCLES (TOTAL: {len(cycles)}) ---")
    for c in cycles[:5]:
        print(f"Cycle {c['cycle_id']} [Len {c['length']}]: {c['cycle_path']} | Vol: ${c['total_cycle_volume_usd']:,.2f} | Risk: {c['risk_score']}")

    if args.export_gephi:
        gephi_path = os.path.join(root, "data", "processed", "crypto_wash_network.graphml")
        analyzer.export_graphml(gephi_path)


if __name__ == "__main__":
    main()
