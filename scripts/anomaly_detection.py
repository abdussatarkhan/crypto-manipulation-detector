"""
Real-Time Statistical Anomaly Detection & Market Manipulation Surveillance
==========================================================================
Implements EWMA (Exponentially Weighted Moving Average) control charts,
sliding-window dynamic Z-score thresholds, order-cancel ratio tracking for
spoofing detection, and bid-ask spread volatility surge monitors.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import setup_logger, load_config, get_project_root, calculate_ewma, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor


class StreamingAnomalyDetector:
    """Streaming statistical surveillance engine for detecting volume spikes, spoofing, and spread manipulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        self.config = config or load_config()
        self.logger = logger or setup_logger("anomaly_detector")
        self.anomaly_cfg = self.config.get("anomaly_detection", {})

        # EWMA Parameters
        ewma_cfg = self.anomaly_cfg.get("ewma", {})
        self.alpha_volume = ewma_cfg.get("alpha_volume", 0.15)
        self.alpha_spread = ewma_cfg.get("alpha_spread", 0.20)
        self.alpha_cancel = ewma_cfg.get("alpha_cancel_ratio", 0.25)
        self.control_sigma = ewma_cfg.get("control_limit_sigma", 3.0)

        # Sliding Window Parameters
        window_cfg = self.anomaly_cfg.get("sliding_window", {})
        self.z_score_threshold = window_cfg.get("z_score_threshold", 2.85)

        # Spoofing Parameters
        spoof_cfg = self.anomaly_cfg.get("spoofing", {})
        self.cancel_ratio_threshold = spoof_cfg.get("cancel_to_trade_ratio_threshold", 8.0)
        self.flash_spread_threshold_bps = spoof_cfg.get("flash_spread_threshold_bps", 12.0)

    def apply_ewma_control_chart(
        self,
        series: pd.Series,
        alpha: float = 0.15,
        sigma_multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Applies an EWMA control chart to a metric series to compute dynamic Upper and Lower Control Limits (UCL/LCL).

        Args:
            series: Metric pandas Series indexed by time.
            alpha: Exponential smoothing factor (0 < alpha <= 1).
            sigma_multiplier: Number of standard deviations for control limits.

        Returns:
            DataFrame with columns: ['value', 'ewma_mean', 'ewma_std', 'ucl', 'lcl', 'is_outlier']
        """
        values = series.to_numpy(dtype=float)
        n = len(values)
        if n == 0:
            return pd.DataFrame()

        ewma_mean, ewma_std = calculate_ewma(values, alpha=alpha)
        ucl = ewma_mean + sigma_multiplier * ewma_std
        lcl = np.maximum(0.0, ewma_mean - sigma_multiplier * ewma_std)
        is_outlier = (values > ucl) | (values < lcl)

        result_df = pd.DataFrame({
            "value": values,
            "ewma_mean": ewma_mean,
            "ewma_std": ewma_std,
            "ucl": ucl,
            "lcl": lcl,
            "is_outlier": is_outlier
        }, index=series.index)
        return result_df

    def detect_volume_spikes(self, bars_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculates dynamic sliding-window Z-scores and EWMA boundaries on trading volume.

        Args:
            bars_df: Aggregated OHLCV DataFrame with 'volume' and 'notional_usd'.
            window: Rolling window size.

        Returns:
            DataFrame containing anomaly markers, z-scores, and probability scores.
        """
        df = bars_df.copy()
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column.")

        # 1. Rolling Z-Score
        rolling_mean = df["volume"].rolling(window=window, min_periods=5).mean()
        rolling_std = df["volume"].rolling(window=window, min_periods=5).std().replace(0, 1e-6)
        df["volume_zscore"] = (df["volume"] - rolling_mean) / rolling_std
        df["volume_zscore"] = df["volume_zscore"].fillna(0.0)

        # 2. EWMA Control Limits
        ewma_res = self.apply_ewma_control_chart(
            df["volume"],
            alpha=self.alpha_volume,
            sigma_multiplier=self.control_sigma
        )
        df["ewma_volume_mean"] = ewma_res["ewma_mean"].values
        df["ewma_volume_ucl"] = ewma_res["ucl"].values
        df["ewma_volume_outlier"] = ewma_res["is_outlier"].values

        # 3. Anomaly Probability Score [0.0, 1.0]
        # Sigmoid-transformed Z-score clamped between 0 and 1
        df["volume_anomaly_score"] = 1.0 / (1.0 + np.exp(-(df["volume_zscore"] - self.z_score_threshold)))
        df["is_volume_anomaly"] = (df["volume_zscore"] > self.z_score_threshold) | df["ewma_volume_outlier"]

        anomaly_count = df["is_volume_anomaly"].sum()
        self.logger.info(f"Volume Anomaly Analysis: Detected {anomaly_count} volume surge events across {len(df)} bars.")
        return df

    def detect_order_book_manipulation(
        self,
        order_events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Monitors high-frequency order cancellations, spoofing signals, and quote stuffing.
        Requires DataFrame with 'placed_orders', 'canceled_orders', 'executed_trades', 'spread_bps'.
        If synthetic, simulates realistic depth dynamics.
        """
        df = order_events.copy()
        if "canceled_orders" not in df.columns or "executed_trades" not in df.columns:
            # Generate representative order book flow metrics
            n = len(df)
            np.random.seed(42)
            trades_cnt = np.random.poisson(lam=15, size=n) + 1
            cancels_cnt = trades_cnt * np.random.uniform(1.2, 3.5, size=n)
            spread_bps = np.random.gamma(shape=2.0, scale=1.5, size=n)

            # Inject spoofing bursts
            spoof_idx = np.random.choice(n, size=max(2, n // 20), replace=False)
            cancels_cnt[spoof_idx] = trades_cnt[spoof_idx] * np.random.uniform(12.0, 28.0, size=len(spoof_idx))
            spread_bps[spoof_idx] *= 3.5

            df["executed_trades"] = trades_cnt
            df["canceled_orders"] = cancels_cnt
            df["spread_bps"] = spread_bps

        # Cancel-to-Trade Ratio
        df["cancel_to_trade_ratio"] = df["canceled_orders"] / np.maximum(df["executed_trades"], 1)

        # EWMA on Cancel Ratio
        ewma_cancel = self.apply_ewma_control_chart(
            df["cancel_to_trade_ratio"],
            alpha=self.alpha_cancel,
            sigma_multiplier=self.control_sigma
        )
        df["ewma_cancel_ucl"] = ewma_cancel["ucl"].values
        df["is_cancel_ratio_anomaly"] = df["cancel_to_trade_ratio"] > np.maximum(self.cancel_ratio_threshold, df["ewma_cancel_ucl"])

        # EWMA on Spread Volatility
        ewma_spread = self.apply_ewma_control_chart(
            df["spread_bps"],
            alpha=self.alpha_spread,
            sigma_multiplier=self.control_sigma
        )
        df["is_spread_anomaly"] = df["spread_bps"] > np.maximum(self.flash_spread_threshold_bps, ewma_spread["ucl"].values)

        # Combined Spoofing Score
        df["spoofing_score"] = np.clip(
            (df["cancel_to_trade_ratio"] / self.cancel_ratio_threshold) * 0.6 +
            (df["spread_bps"] / self.flash_spread_threshold_bps) * 0.4,
            0.0, 1.0
        )
        df["is_spoofing_alert"] = df["is_cancel_ratio_anomaly"] & (df["spoofing_score"] > 0.65)

        spoof_count = df["is_spoofing_alert"].sum()
        self.logger.info(f"Order-Book Surveillance: Identified {spoof_count} suspicious spoofing/layering intervals.")
        return df

    def run_full_pipeline(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Executes full anomaly detection across volume, order dynamics, and spreads."""
        self.logger.info("Executing comprehensive anomaly surveillance pipeline...")
        df = self.detect_volume_spikes(bars_df)
        df = self.detect_order_book_manipulation(df)
        return df


def main():
    parser = argparse.ArgumentParser(description="Statistical anomaly detection for trade volumes and order flows.")
    parser.add_argument("--bars", type=str, default=None, help="Path to resampled OHLCV bars parquet")
    args = parser.parse_args()

    root = get_project_root()
    detector = StreamingAnomalyDetector()

    if args.bars and os.path.exists(args.bars):
        bars_df = pd.read_parquet(args.bars) if args.bars.endswith(".parquet") else pd.read_csv(args.bars)
    else:
        detector.logger.info("Generating synthetic bar sequence for testing...")
        synthetic_stream = generate_synthetic_trade_stream(num_trades=1000, inject_wash_trading=True)
        prep = TradePreprocessor()
        clean_trades = prep.normalize_trade_dataframe(synthetic_stream)
        bars_df = prep.aggregate_to_bars(clean_trades, freq="1min")

    surveillance_df = detector.run_full_pipeline(bars_df)
    anomalies = surveillance_df[surveillance_df["is_volume_anomaly"] | surveillance_df["is_spoofing_alert"]]

    print("\n--- DETECTED MARKET ANOMALIES ---")
    if not anomalies.empty:
        cols = ["datetime_utc", "close", "volume", "volume_zscore", "cancel_to_trade_ratio", "spoofing_score", "is_volume_anomaly"]
        available_cols = [c for c in cols if c in anomalies.columns]
        print(anomalies[available_cols].head(10).to_string(index=False))
    else:
        print("No anomalous events triggered at current threshold parameters.")

    out_path = os.path.join(root, "data", "processed", "surveillance_anomalies.parquet")
    surveillance_df.to_parquet(out_path, index=False)
    detector.logger.info(f"Saved complete anomaly inspection results to: {out_path}")


if __name__ == "__main__":
    main()
