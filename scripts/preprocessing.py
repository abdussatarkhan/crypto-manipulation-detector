"""
Trade Data Preprocessing & Wallet Normalization Pipeline
========================================================
Performs high-speed deduplication, timestamp normalization, exchange entity
wallet attribution, tick-to-bar aggregation, and price outlier cleaning.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import setup_logger, load_config, get_project_root, generate_synthetic_trade_stream

# Known crypto exchange wallet clusters for attribution
KNOWN_EXCHANGE_WALLETS = {
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": {"entity": "Binance", "type": "Hot Wallet"},
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": {"entity": "Binance", "type": "Cold Storage"},
    "0x564286362092d8e7936f0549571a803b203aaced": {"entity": "Binance", "type": "Deposit Wallet"},
    "0x28c6c06298d514db089934071355e5743bf21d60": {"entity": "Binance", "type": "Pay In"},
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": {"entity": "Coinbase", "type": "Hot Wallet"},
    "0x503828976d22510aad0201ac7ec88293211d23dc": {"entity": "Coinbase", "type": "Cold Storage"},
    "0xab5c667526405d123280680302956acbdfc738b5": {"entity": "Kraken", "type": "Hot Wallet"},
    "0x2faf487a4414fe30a23f669a4f73a02f7acc226e": {"entity": "Kraken", "type": "Deposit Pool"},
    "0x1111111254fb6c44bac0bed2854e76f90643097d": {"entity": "1inch DEX", "type": "Aggregation Router"},
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": {"entity": "Uniswap", "type": "Swap Router V3"},
}


class TradePreprocessor:
    """Preprocesses raw high-frequency trades into research-grade analytics datasets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        self.config = config or load_config()
        self.logger = logger or setup_logger("preprocessor")
        self.prep_config = self.config.get("preprocessing", {})
        self.outlier_pct = self.prep_config.get("outlier_price_filter_pct", 0.20)

    def map_wallet_entity(self, address: str) -> Dict[str, str]:
        """
        Maps an on-chain or internal wallet address to a known exchange or entity.

        Args:
            address: Wallet address string.

        Returns:
            Dict containing 'entity' name and 'type' classification.
        """
        if not address or pd.isna(address):
            return {"entity": "Unknown", "type": "Unassigned"}

        clean_addr = str(address).lower().strip()
        if clean_addr in KNOWN_EXCHANGE_WALLETS:
            return KNOWN_EXCHANGE_WALLETS[clean_addr]

        if clean_addr.startswith("0xwash"):
            return {"entity": "Suspected Wash Syndicate", "type": "Syndicate Node"}
        elif "binance" in clean_addr:
            return {"entity": "Binance Internal", "type": "Matching Engine"}
        elif clean_addr.startswith("0xmaker") or clean_addr.startswith("0xtaker"):
            return {"entity": "Market Participant", "type": "Regular Trader"}
        else:
            return {"entity": "External Private Wallet", "type": "Retail/Prop"}

    def deduplicate_trades(self, df: pd.DataFrame, id_col: str = "trade_id") -> pd.DataFrame:
        """Removes duplicate trade executions based on trade ID."""
        initial_len = len(df)
        if id_col in df.columns:
            df = df.drop_duplicates(subset=[id_col], keep="first")
        else:
            df = df.drop_duplicates(subset=["timestamp", "price", "quantity"], keep="first")
        removed = initial_len - len(df)
        if removed > 0:
            self.logger.info(f"Deduplication: Dropped {removed} redundant trade rows.")
        return df

    def filter_price_outliers(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Filters flash crashes, bad wicks, and pricing errors exceeding outlier_pct
        relative to the local rolling median.
        """
        if len(df) < window:
            return df

        rolling_median = df["price"].rolling(window=window, min_periods=5, center=True).median()
        deviation = (df["price"] - rolling_median).abs() / (rolling_median + 1e-8)
        mask = deviation <= self.outlier_pct
        filtered_df = df[mask].copy()
        outliers_count = len(df) - len(filtered_df)
        if outliers_count > 0:
            self.logger.warning(f"Filtered {outliers_count} price wick outlier trades (> {self.outlier_pct*100:.0f}% deviation).")
        return filtered_df

    def normalize_trade_dataframe(self, raw_data: Any) -> pd.DataFrame:
        """
        Transforms raw trade dictionaries or raw DataFrame into standardized schema.

        Schema:
            trade_id, timestamp_ms, datetime_utc, symbol, price, quantity,
            notional_usd, buyer_address, seller_address, buyer_entity, seller_entity,
            is_buyer_maker, is_inter_exchange
        """
        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        else:
            raise ValueError("Input must be a list of trade dicts or a pandas DataFrame.")

        # Harmonize column naming from Binance formats
        col_mappings = {
            "t": "trade_id",
            "a": "trade_id",
            "agg_trade_id": "trade_id",
            "T": "timestamp_ms",
            "E": "timestamp_ms",
            "timestamp": "timestamp_ms",
            "p": "price",
            "q": "quantity",
            "s": "symbol",
            "b": "buyer_address",
            "a": "seller_address",
            "m": "is_buyer_maker"
        }
        for old_col, new_col in col_mappings.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # Fill missing required columns
        if "trade_id" not in df.columns:
            df["trade_id"] = range(1, len(df) + 1)
        if "timestamp_ms" not in df.columns:
            df["timestamp_ms"] = int(time.time() * 1000)
        if "symbol" not in df.columns:
            df["symbol"] = "BTCUSDT"
        if "buyer_address" not in df.columns:
            df["buyer_address"] = "0xMaker"
        if "seller_address" not in df.columns:
            df["seller_address"] = "0xTaker"
        if "is_buyer_maker" not in df.columns:
            df["is_buyer_maker"] = False

        # Numeric conversions
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("int64")
        df.dropna(subset=["price", "quantity", "timestamp_ms"], inplace=True)

        # Value and time computations
        df["notional_usd"] = df["price"] * df["quantity"]
        df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df.sort_values("timestamp_ms", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Deduplicate & Filter
        df = self.deduplicate_trades(df)
        df = self.filter_price_outliers(df)

        # Entity attribution
        buyer_meta = [self.map_wallet_entity(addr) for addr in df["buyer_address"]]
        seller_meta = [self.map_wallet_entity(addr) for addr in df["seller_address"]]

        df["buyer_entity"] = [m["entity"] for m in buyer_meta]
        df["buyer_type"] = [m["type"] for m in buyer_meta]
        df["seller_entity"] = [m["entity"] for m in seller_meta]
        df["seller_type"] = [m["type"] for m in seller_meta]

        df["is_inter_exchange"] = (df["buyer_entity"] != df["seller_entity"]) & \
                                  (df["buyer_entity"] != "External Private Wallet") & \
                                  (df["seller_entity"] != "External Private Wallet")

        self.logger.info(f"Successfully normalized dataset: {len(df)} cleaned trades.")
        return df

    def aggregate_to_bars(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """
        Resamples cleaned trade stream into OHLCV bars with buy/sell taker volumes.
        """
        if df.empty:
            return pd.DataFrame()

        df_indexed = df.set_index("datetime_utc").sort_index()

        bars = df_indexed["price"].resample(freq).ohlc()
        bars["volume"] = df_indexed["quantity"].resample(freq).sum()
        bars["notional_usd"] = df_indexed["notional_usd"].resample(freq).sum()
        bars["trade_count"] = df_indexed["trade_id"].resample(freq).count()

        # Taker buy flow
        taker_buys = df_indexed[df_indexed["is_buyer_maker"] == False]["quantity"].resample(freq).sum()
        bars["taker_buy_vol"] = taker_buys.reindex(bars.index).fillna(0.0)
        bars["taker_sell_vol"] = bars["volume"] - bars["taker_buy_vol"]
        bars["volume_imbalance"] = (bars["taker_buy_vol"] - bars["taker_sell_vol"]) / (bars["volume"] + 1e-9)

        bars.dropna(subset=["close"], inplace=True)
        bars.reset_index(inplace=True)
        return bars


def main():
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and normalize crypto trade data.")
    parser.add_argument("--input", type=str, default=None, help="Input raw CSV or Parquet file path")
    parser.add_argument("--output", type=str, default=None, help="Output destination for cleaned dataset")
    parser.add_argument("--resample", type=str, default="1min", help="Resampling bar frequency (e.g. 1min, 5min)")
    args = parser.parse_args()

    root = get_project_root()
    preprocessor = TradePreprocessor()

    if args.input and os.path.exists(args.input):
        if args.input.endswith(".parquet"):
            raw_data = pd.read_parquet(args.input)
        else:
            raw_data = pd.read_csv(args.input)
        preprocessor.logger.info(f"Loaded {len(raw_data)} records from {args.input}")
    else:
        preprocessor.logger.info("No input file provided or file not found. Generating synthetic stream for demonstration.")
        raw_data = generate_synthetic_trade_stream(num_trades=600, inject_wash_trading=True)

    cleaned_df = preprocessor.normalize_trade_dataframe(raw_data)
    bars_df = preprocessor.aggregate_to_bars(cleaned_df, freq=args.resample)

    out_dir = os.path.join(root, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    clean_path = args.output or os.path.join(out_dir, "cleaned_trades.parquet")
    cleaned_df.to_parquet(clean_path, index=False)
    preprocessor.logger.info(f"Saved normalized trades to: {clean_path}")

    bars_path = os.path.join(out_dir, f"ohlcv_bars_{args.resample}.parquet")
    bars_df.to_parquet(bars_path, index=False)
    preprocessor.logger.info(f"Saved resampled {args.resample} bars to: {bars_path}")


if __name__ == "__main__":
    main()
