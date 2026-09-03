"""
Composite Market Manipulation Alert & Probability Scoring Engine
================================================================
Combines multi-modal signals—counterparty graph topology, EWMA statistical
anomalies, Benford's Law forensic deviations, and order-book spoofing metrics—
into a unified Manipulation Probability Index (MPI) and generates regulatory alerts.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import setup_logger, load_config, get_project_root, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.graph_analysis import CryptoGraphAnalyzer
from scripts.anomaly_detection import StreamingAnomalyDetector
from scripts.benfords_law import BenfordsLawAnalyzer


class CompositeAlertEngine:
    """Multi-modal surveillance and manipulation scoring engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        self.config = config or load_config()
        self.logger = logger or setup_logger("composite_scorer")
        scoring_cfg = self.config.get("alert_scoring", {})

        # Weights
        weights = scoring_cfg.get("weights", {})
        self.w_graph = weights.get("graph_cycle", 0.35)
        self.w_volume = weights.get("volume_ewma", 0.25)
        self.w_benford = weights.get("benford_anomaly", 0.20)
        self.w_spoof = weights.get("order_cancel_spoofing", 0.20)

        # Risk thresholds
        thresh = scoring_cfg.get("thresholds", {})
        self.thresh_low = thresh.get("low_risk", 0.35)
        self.thresh_med = thresh.get("medium_risk", 0.65)
        self.thresh_high = thresh.get("high_risk", 0.85)

        # Sub-analyzers
        self.graph_analyzer = CryptoGraphAnalyzer(config=self.config, logger=self.logger)
        self.anomaly_detector = StreamingAnomalyDetector(config=self.config, logger=self.logger)
        self.benford_analyzer = BenfordsLawAnalyzer(config=self.config, logger=self.logger)

    def calculate_composite_score(
        self,
        graph_score: float,
        volume_score: float,
        benford_score: float,
        spoof_score: float
    ) -> Tuple[float, str]:
        """
        Computes the weighted composite Manipulation Probability Index (MPI)
        and classifies into risk tiers.

        Returns:
            Tuple of (composite_score [0.0 - 1.0], severity_tier)
        """
        mpi = (
            self.w_graph * np.clip(graph_score, 0.0, 1.0) +
            self.w_volume * np.clip(volume_score, 0.0, 1.0) +
            self.w_benford * np.clip(benford_score, 0.0, 1.0) +
            self.w_spoof * np.clip(spoof_score, 0.0, 1.0)
        )
        mpi = round(float(np.clip(mpi, 0.0, 1.0)), 4)

        if mpi >= self.thresh_high:
            tier = "CRITICAL"
        elif mpi >= self.thresh_med:
            tier = "HIGH"
        elif mpi >= self.thresh_low:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return mpi, tier

    def evaluate_dataset(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes end-to-end multi-modal scoring on a trade dataset.

        Args:
            trades_df: Cleaned trade records DataFrame.

        Returns:
            Comprehensive audit report dictionary with all signals, scores, and active alerts.
        """
        self.logger.info(f"Initiating composite evaluation on {len(trades_df)} trade events...")

        # 1. Graph Analysis
        self.graph_analyzer.build_transaction_graph(trades_df)
        cycles = self.graph_analyzer.detect_wash_trading_cycles()
        communities = self.graph_analyzer.run_community_detection()
        centrality_df = self.graph_analyzer.compute_centrality_metrics()

        # Aggregate graph signal
        max_cycle_risk = max([c["risk_score"] for c in cycles]) if cycles else 0.0
        syndicate_max = centrality_df["syndicate_orchestrator_score"].max() if not centrality_df.empty else 0.0
        graph_signal = min(1.0, 0.6 * max_cycle_risk + 0.4 * syndicate_max)

        # 2. Benford's Law Forensic Analysis
        benford_res = self.benford_analyzer.analyze_distribution(trades_df["quantity"], series_name="Trade Quantities")
        benford_signal = benford_res.get("manipulation_probability", 0.0)

        # 3. Statistical Anomaly & Order Dynamics
        preprocessor = TradePreprocessor(self.config)
        bars_df = preprocessor.aggregate_to_bars(trades_df, freq="1min")
        surveillance_df = self.anomaly_detector.run_full_pipeline(bars_df)

        volume_signal = surveillance_df["volume_anomaly_score"].mean() if not surveillance_df.empty else 0.0
        volume_signal = min(1.0, volume_signal * 1.5)  # Scale up sensitivity
        spoof_signal = surveillance_df["spoofing_score"].max() if not surveillance_df.empty else 0.0

        # 4. Compute Master Composite Manipulation Probability Index (MPI)
        composite_mpi, severity = self.calculate_composite_score(
            graph_score=graph_signal,
            volume_score=volume_signal,
            benford_score=benford_signal,
            spoof_score=spoof_signal
        )

        # 5. Synthesize Actionable Compliance Alerts
        alerts = []
        alert_id_counter = 1

        # Wash Cycle Alerts
        for cycle in cycles[:10]:
            alerts.append({
                "alert_id": f"ALT-GR-{alert_id_counter:04d}",
                "timestamp": int(time.time() * 1000),
                "type": "WASH_TRADING_RING",
                "severity": "CRITICAL" if cycle["is_self_trade"] or cycle["risk_score"] > 0.85 else "HIGH",
                "score": cycle["risk_score"],
                "description": f"Detected circular trade loop: {cycle['cycle_path']}",
                "volume_usd": cycle["total_cycle_volume_usd"],
                "entities_involved": cycle["nodes"]
            })
            alert_id_counter += 1

        # Benford Non-Conformity Alert
        if benford_signal >= 0.50:
            alerts.append({
                "alert_id": f"ALT-BN-{alert_id_counter:04d}",
                "timestamp": int(time.time() * 1000),
                "type": "BENFORD_DISTRIBUTION_ANOMALY",
                "severity": "HIGH" if benford_signal >= 0.70 else "MEDIUM",
                "score": benford_signal,
                "description": f"Trade size distribution violates Benford's Law (MAD={benford_res['mad']}, KS Reject={benford_res['ks_rejected']})",
                "volume_usd": trades_df["notional_usd"].sum(),
                "entities_involved": ["Market-Wide Ingestion"]
            })
            alert_id_counter += 1

        # Volume Surge Alerts
        vol_anomalies = surveillance_df[surveillance_df["is_volume_anomaly"]]
        for _, row in vol_anomalies.head(5).iterrows():
            alerts.append({
                "alert_id": f"ALT-VL-{alert_id_counter:04d}",
                "timestamp": int(pd.to_datetime(row["datetime_utc"]).timestamp() * 1000),
                "type": "VOLUME_SPIKE_OUTLIER",
                "severity": "HIGH" if row["volume_zscore"] > 4.0 else "MEDIUM",
                "score": round(min(1.0, row["volume_zscore"] / 5.0), 3),
                "description": f"Volume spike detected (Z-Score: {row['volume_zscore']:.2f}, EWMA UCL breached)",
                "volume_usd": round(row.get("notional_usd", 0.0), 2),
                "entities_involved": ["Order Book Liquidity Pool"]
            })
            alert_id_counter += 1

        summary = {
            "evaluation_timestamp": int(time.time() * 1000),
            "symbol": trades_df["symbol"].iloc[0] if "symbol" in trades_df.columns else "MULTI",
            "total_trades_analyzed": len(trades_df),
            "total_volume_usd": round(float(trades_df["notional_usd"].sum()), 2),
            "manipulation_probability_index": composite_mpi,
            "overall_severity": severity,
            "sub_signals": {
                "graph_cycle_score": round(graph_signal, 4),
                "volume_anomaly_score": round(volume_signal, 4),
                "benfords_law_score": round(benford_signal, 4),
                "spoofing_order_score": round(spoof_signal, 4)
            },
            "weights_used": {
                "graph": self.w_graph,
                "volume": self.w_volume,
                "benford": self.w_benford,
                "spoofing": self.w_spoof
            },
            "active_alerts_count": len(alerts),
            "alerts": alerts,
            "top_suspect_wallets": centrality_df.head(5).to_dict(orient="records") if not centrality_df.empty else []
        }

        self.logger.info(
            f"Evaluation Complete: Composite MPI = {composite_mpi:.3f} [{severity}] | "
            f"Active Alerts = {len(alerts)}"
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-modal composite manipulation scoring engine.")
    parser.add_argument("--input", type=str, default=None, help="Input trade parquet file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON alert report path")
    args = parser.parse_args()

    root = get_project_root()
    engine = CompositeAlertEngine()

    if args.input and os.path.exists(args.input):
        trades_df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    else:
        engine.logger.info("Generating synthetic trade dataset with injected wash-trading patterns...")
        raw_stream = generate_synthetic_trade_stream(num_trades=800, inject_wash_trading=True)
        prep = TradePreprocessor()
        trades_df = prep.normalize_trade_dataframe(raw_stream)

    report = engine.evaluate_dataset(trades_df)

    print("\n========================================================")
    print("      COMPOSITE CRYPTO MANIPULATION AUDIT REPORT        ")
    print("========================================================")
    print(f"Overall Manipulation Probability Index: {report['manipulation_probability_index']*100:.1f}%")
    print(f"Regulatory Risk Severity Tier:          {report['overall_severity']}")
    print(f"Total Trades Examined:                  {report['total_trades_analyzed']}")
    print(f"Total Market Volume (USD):              ${report['total_volume_usd']:,.2f}")
    print("--------------------------------------------------------")
    print("Sub-Component Signal Scores:")
    for sig_name, val in report["sub_signals"].items():
        print(f"  - {sig_name:<28}: {val*100:5.1f}%")
    print("--------------------------------------------------------")
    print(f"Total High-Priority Alerts Raised:      {report['active_alerts_count']}")
    for a in report["alerts"][:5]:
        print(f"  * [{a['severity']}] {a['type']}: {a['description']}")
    print("========================================================\n")

    out_path = args.output or os.path.join(root, "data", "processed", "composite_manipulation_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    engine.logger.info(f"Saved complete audit report to: {out_path}")


if __name__ == "__main__":
    main()
