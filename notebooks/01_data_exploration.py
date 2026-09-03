# %% [markdown]
# # Notebook 01: Exploratory Data Analysis & Trade Flow Dynamics
# 
# **Project**: Real-Time Crypto Wash-Trade & Market Manipulation Detector  
# **Domain**: Finance / FinTech / High-Frequency Market Surveillance  
# **Objective**: Inspect raw cryptocurrency trade data, explore tick-to-bar aggregation,
# analyze inter-arrival time distributions, and benchmark volume profiles across trading counterparties.

# %%
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Ensure project root is accessible
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import load_config, setup_logger, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.data_collection_historical import BinanceHistoricalClient

logger = setup_logger("nb_01_exploration")
config = load_config()
print("Configuration loaded successfully. Default trading symbols:", config["ingestion"]["binance"]["symbols"])

# %% [markdown]
# ## 1. Ingest Raw Trade Stream
# 
# We pull or generate a high-frequency stream of tick trades, including ground-truth injected wash trades
# and volume bursts to evaluate baseline distribution patterns.

# %%
# Generate realistic trade stream with injected wash cycles and volume anomalies
num_trades = 1200
raw_stream = generate_synthetic_trade_stream(
    symbol="BTCUSDT",
    num_trades=num_trades,
    base_price=64250.0,
    inject_wash_trading=True,
    random_seed=42
)

raw_df = pd.DataFrame(raw_stream)
print(f"Total raw trade ticks ingested: {len(raw_df)}")
display_cols = ["t", "T", "s", "p", "q", "b", "a", "m", "anomaly_label"]
print(raw_df[display_cols].head(10))

# %% [markdown]
# ## 2. Data Preprocessing & Wallet Entity Normalization
# 
# Standardize column headers, convert epoch timestamps, filter flash-crash price spikes,
# and attribute addresses to known exchange entities or suspicious syndicate wallets.

# %%
preprocessor = TradePreprocessor(config=config)
cleaned_df = preprocessor.normalize_trade_dataframe(raw_df)

print(f"Cleaned trades count: {len(cleaned_df)}")
print("\nDataset Info:")
print(cleaned_df.info())

print("\nSample Processed Trades:")
print(cleaned_df[["trade_id", "datetime_utc", "symbol", "price", "quantity", "notional_usd", "buyer_entity", "seller_entity"]].head())

# %% [markdown]
# ## 3. Trade Size & Notional Volume Distribution
# 
# Natural crypto trade sizes typically follow a heavy-tailed, log-normal distribution.
# Artificial wash trades cluster around specific artificial quantities.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trade Size (Quantity) Distribution
sns.histplot(cleaned_df["quantity"], bins=50, kde=True, ax=axes[0], color="royalblue")
axes[0].set_title("Distribution of Trade Sizes (BTC)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Quantity")
axes[0].set_ylabel("Frequency")
axes[0].set_yscale("log")

# Notional Value (USD) Distribution
sns.histplot(cleaned_df["notional_usd"], bins=50, kde=True, ax=axes[1], color="forestgreen")
axes[1].set_title("Distribution of Notional Value (USD)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("USD Value")
axes[1].set_ylabel("Frequency")
axes[1].set_yscale("log")

plt.tight_layout()
plt.show()

# Summary statistics
print("\n--- NOTIONAL VALUE SUMMARY (USD) ---")
print(cleaned_df["notional_usd"].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.99]))

# %% [markdown]
# ## 4. Inter-Arrival Time Analysis
# 
# Organic market participants exhibit Poisson-process arrival times with exponential inter-arrival intervals.
# High-frequency robotic wash trading often clusters at sub-millisecond intervals with fixed periodicity.

# %%
cleaned_df["time_diff_ms"] = cleaned_df["timestamp_ms"].diff().fillna(0)

plt.figure(figsize=(10, 5))
sns.histplot(cleaned_df["time_diff_ms"], bins=40, kde=True, color="darkorange")
plt.title("Inter-Trade Arrival Time Distribution (Milliseconds)", fontsize=13, fontweight="bold")
plt.xlabel("Delta Time Between Consecutive Trades (ms)")
plt.ylabel("Frequency")
plt.show()

print(f"Mean trade arrival interval: {cleaned_df['time_diff_ms'].mean():.2f} ms")
print(f"Median trade arrival interval: {cleaned_df['time_diff_ms'].median():.2f} ms")
print(f"Standard deviation: {cleaned_df['time_diff_ms'].std():.2f} ms")

# %% [markdown]
# ## 5. Aggregation into OHLCV Candlestick Bars
# 
# Aggregate raw tick executions into 1-minute time bars to analyze buy/sell volume imbalances.

# %%
bars_df = preprocessor.aggregate_to_bars(cleaned_df, freq="1min")
print(f"Aggregated into {len(bars_df)} OHLCV bars.")
print(bars_df.head())

# Plot Price and Volume Flow
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(bars_df["datetime_utc"], bars_df["close"], label="Close Price", color="cyan", linewidth=1.5)
ax1.set_title("BTC/USDT OHLCV Aggregation & Taker Flow", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")

# Bar Volume with Taker Imbalance
colors = ["red" if row["volume_imbalance"] < 0 else "green" for _, row in bars_df.iterrows()]
ax2.bar(bars_df["datetime_utc"], bars_df["volume"], color=colors, alpha=0.7, width=0.0005)
ax2.set_ylabel("Volume (BTC)")
ax2.set_xlabel("Time (UTC)")

plt.tight_layout()
plt.show()
print("Exploratory Data Analysis Complete.")
