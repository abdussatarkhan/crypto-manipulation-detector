"""
Utility Functions for Crypto Manipulation Detector
===================================================
Provides logging configuration, YAML configuration parsing, simulated streaming generators,
Kafka client helpers with graceful fallbacks, and mathematical helpers.
"""

import os
import sys
import time
import logging
import datetime
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def get_project_root() -> str:
    """Returns the absolute root path of the project repository."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration from file.

    Args:
        config_path: Path to the YAML configuration file. If None, resolves to config/config.yaml.

    Returns:
        Dict containing configuration parameters.
    """
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "crypto_detector", log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Configures and returns a structured logger with standard formatting.

    Args:
        name: Logger identifier.
        log_file: Optional file destination for logs.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logging.Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers on re-instantiation
    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger


def format_timestamp_ms(timestamp_ms: int) -> str:
    """Converts a millisecond epoch timestamp to ISO 8601 string."""
    dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def parse_timestamp_iso(iso_str: str) -> int:
    """Converts an ISO 8601 string to millisecond epoch timestamp."""
    dt = pd.to_datetime(iso_str, utc=True)
    return int(dt.timestamp() * 1000)


def generate_synthetic_trade_stream(
    symbol: str = "BTCUSDT",
    num_trades: int = 500,
    base_price: float = 65000.0,
    inject_wash_trading: bool = True,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates synthetic trade ticks conforming to Binance WebSocket trade format,
    optionally injecting coordinated wash-trading cycles and volume anomalies.

    Args:
        symbol: Trading pair ticker.
        num_trades: Total number of trades to simulate.
        base_price: Initial reference price.
        inject_wash_trading: If True, injects cyclical wallet trade patterns and volume spikes.
        random_seed: Seed for reproducibility.

    Returns:
        List of trade dictionaries.
    """
    np.random.seed(random_seed)
    trades = []
    current_time_ms = int(time.time() * 1000) - (num_trades * 100)
    current_price = base_price

    # Wallets pool
    regular_wallets = [f"0x{i:04x}{i*7:04x}abcdef12345678" for i in range(1, 30)]
    syndicate_wallets = [f"0xWASH_{i:02d}_9999888877776666" for i in range(1, 6)]

    trade_id = 1000000

    for i in range(num_trades):
        current_time_ms += int(np.random.exponential(scale=120) + 10)
        price_delta = np.random.normal(0, base_price * 0.0003)
        current_price = max(base_price * 0.5, current_price + price_delta)

        # Baseline volume: log-normal
        volume = float(np.random.lognormal(mean=-1.5, sigma=1.0))
        volume = max(0.001, round(volume, 4))

        buyer = np.random.choice(regular_wallets)
        seller = np.random.choice(regular_wallets)
        while seller == buyer:
            seller = np.random.choice(regular_wallets)

        is_wash = False
        anomaly_flag = "NONE"

        # Inject Wash Trading Ring (Every ~40 trades)
        if inject_wash_trading and (i % 40 < 5):
            is_wash = True
            ring_step = i % 40
            buyer = syndicate_wallets[ring_step % len(syndicate_wallets)]
            seller = syndicate_wallets[(ring_step + 1) % len(syndicate_wallets)]
            # In wash trading, trade sizes repeat or deviate from Benford's Law (e.g. repeated 7.777, 9.999)
            volume = float(np.random.choice([7.777, 8.888, 9.999, 5.555, 4.444]))
            anomaly_flag = "WASH_RING_CYCLE"

        # Inject Volume Spike Anomaly
        elif inject_wash_trading and (i in [120, 121, 122, 280, 281]):
            volume = float(np.random.uniform(50.0, 150.0))  # 50x-150x regular size
            anomaly_flag = "VOLUME_SPIKE"

        trade_id += 1
        trades.append({
            "e": "trade",
            "E": current_time_ms,
            "s": symbol.upper(),
            "t": trade_id,
            "p": f"{current_price:.2f}",
            "q": f"{volume:.4f}",
            "b": buyer,
            "a": seller,
            "T": current_time_ms,
            "m": bool(np.random.choice([True, False])),
            "M": True,
            "is_wash_ground_truth": is_wash,
            "anomaly_label": anomaly_flag
        })

    return trades


def compute_rolling_zscores(series: pd.Series, window: int = 30) -> pd.Series:
    """Computes sliding-window z-score for a pandas Series."""
    rolling_mean = series.rolling(window=window, min_periods=max(5, window // 4)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(5, window // 4)).std()
    z_scores = (series - rolling_mean) / (rolling_std + 1e-9)
    return z_scores.fillna(0.0)


def calculate_ewma(values: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Exponential Weighted Moving Average (EWMA) and dynamic variance.

    Args:
        values: 1D array of metric values.
        alpha: Smoothing factor between 0 and 1.

    Returns:
        Tuple of (ewma_mean_array, ewma_std_array).
    """
    n = len(values)
    ewma_mean = np.zeros(n)
    ewma_var = np.zeros(n)

    if n == 0:
        return ewma_mean, ewma_var

    ewma_mean[0] = values[0]
    ewma_var[0] = 0.0

    for t in range(1, n):
        diff = values[t] - ewma_mean[t-1]
        ewma_mean[t] = ewma_mean[t-1] + alpha * diff
        # Incrementally update variance estimation
        ewma_var[t] = (1 - alpha) * (ewma_var[t-1] + alpha * (diff ** 2))

    ewma_std = np.sqrt(np.maximum(ewma_var, 1e-12))
    return ewma_mean, ewma_std
