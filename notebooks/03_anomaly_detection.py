# %% [markdown]
# # Notebook 03: Streaming Statistical Anomaly Detection & Surveillance
# 
# **Project**: Real-Time Crypto Wash-Trade & Market Manipulation Detector  
# **Objective**: Deploy EWMA (Exponentially Weighted Moving Average) statistical control charts,
# compute dynamic sliding-window Z-score thresholds on volume, and monitor order-cancel ratios
# for spoofing and quote stuffing detection.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import load_config, setup_logger, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.anomaly_detection import StreamingAnomalyDetector

config = load_config()
logger = setup_logger("nb_03_anomaly")
print("Anomaly Detection Environment configured.")

# %% [markdown]
# ## 1. Ingest Trade Data & Aggregate to Time Bars
# 
# We simulate a high-speed trade feed with multiple injected volume spikes and spoofing events.

# %%
stream = generate_synthetic_trade_stream(num_trades=1500, inject_wash_trading=True, random_seed=123)
preprocessor = TradePreprocessor(config=config)
clean_trades = preprocessor.normalize_trade_dataframe(stream)
bars_df = preprocessor.aggregate_to_bars(clean_trades, freq="1min")

print(f"Generated {len(bars_df)} 1-minute time bars.")
print(bars_df.head())

# %% [markdown]
# ## 2. Apply EWMA Control Charts & Sliding-Window Z-Scores
# 
# ### Mathematical Formulations:
# 1. **Exponentially Weighted Moving Average (EWMA)**:
#    $$\text{EWMA}_t = \lambda X_t + (1 - \lambda) \text{EWMA}_{t-1}$$
#    where $\lambda \in (0, 1]$ is the smoothing factor (typically $\lambda = 0.2$ for high-frequency quote feeds).
# 
# 2. **Sliding-Window Volume Z-Score**:
#    $$Z_t = \frac{V_t - \mu_{w}}{\sigma_{w}}$$
#    where $\mu_w$ and $\sigma_w$ represent rolling 15-minute window mean and standard deviation.
#    Volume anomalies exceeding $Z_t > 3.5$ trigger potential spoofing / wash-trade alerts.
# 
# 3. **Order-to-Cancel (OTC) Ratio**:
#    $$\text{OTC}_w = \frac{N_{\text{cancelled}}}{N_{\text{placed}}}$$
#    Alert condition: $\text{OTC}_w > 0.95$ combined with median order lifetime $< 450\text{ms}$.
# 
# The EWMA filter gives exponential weight to recent observations while tracking variance dynamically:
# $$z_t = \lambda x_t + (1 - \lambda) z_{t-1}$$
# Control limits are established at $\mu_t \pm 3 \sigma_t$.

# %%
detector = StreamingAnomalyDetector(config=config)
surveillance_df = detector.run_full_pipeline(bars_df)

print(f"Total bars analyzed: {len(surveillance_df)}")
print("Surveillance Columns:", surveillance_df.columns.tolist())

# Inspect detected volume anomalies
vol_anomalies = surveillance_df[surveillance_df["is_volume_anomaly"]]
print(f"\nDetected {len(vol_anomalies)} volume spike anomalies:")
print(vol_anomalies[["datetime_utc", "volume", "volume_zscore", "ewma_volume_ucl", "volume_anomaly_score"]].head())

# %% [markdown]
# ## 3. Visualizing EWMA Control Charts & Volume Outliers
# 
# Plot the actual bar volumes against the dynamic Upper Control Limit (UCL) and highlight breach points.

# %%
plt.figure(figsize=(14, 6))

x = surveillance_df["datetime_utc"]
plt.plot(x, surveillance_df["volume"], label="Actual Volume (BTC)", color="#00e5ff", linewidth=1.5)
plt.plot(x, surveillance_df["ewma_volume_mean"], label="EWMA Mean (alpha=0.15)", color="#ff9100", linestyle="--", linewidth=1.8)
plt.plot(x, surveillance_df["ewma_volume_ucl"], label="Upper Control Limit (+3 Sigma)", color="#ff1744", linestyle=":", linewidth=2)

# Scatter plot anomaly points
if not vol_anomalies.empty:
    plt.scatter(
        vol_anomalies["datetime_utc"],
        vol_anomalies["volume"],
        color="crimson",
        s=80,
        zorder=5,
        label=f"Volume Outliers (n={len(vol_anomalies)})",
        edgecolor="white"
    )

plt.title("Real-Time EWMA Volume Surveillance & Dynamic Control Limit Breaches", fontsize=14, fontweight="bold")
plt.xlabel("Time (UTC)")
plt.ylabel("Volume (BTC)")
plt.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## 4. Spoofing & Order Book Cancellation Dynamics
# 
# Spoofing is characterized by an extreme ratio of canceled orders to filled executions,
# accompanied by brief artificial bid-ask spread expansions.

# %%
spoof_alerts = surveillance_df[surveillance_df["is_spoofing_alert"]]
print(f"\nDetected {len(spoof_alerts)} spoofing / quote stuffing intervals:")
print(spoof_alerts[["datetime_utc", "cancel_to_trade_ratio", "ewma_cancel_ucl", "spread_bps", "spoofing_score"]].head())

# Plot Order Cancel Ratio
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(x, surveillance_df["cancel_to_trade_ratio"], color="purple", label="Cancel-to-Trade Ratio")
ax1.plot(x, surveillance_df["ewma_cancel_ucl"], color="red", linestyle="--", label="Cancel Ratio UCL")
ax1.axhline(y=config["anomaly_detection"]["spoofing"]["cancel_to_trade_ratio_threshold"], color="orange", linestyle=":", label="Static Threshold (8.0)")
ax1.set_title("Order-Book Cancellation Ratio Surveillance", fontsize=13, fontweight="bold")
ax1.set_ylabel("Ratio")
ax1.legend(loc="upper left")

ax2.plot(x, surveillance_df["spread_bps"], color="teal", label="Bid-Ask Spread (bps)")
ax2.axhline(y=config["anomaly_detection"]["spoofing"]["flash_spread_threshold_bps"], color="red", linestyle=":", label="Spread Flash Threshold (12 bps)")
ax2.set_title("Bid-Ask Spread Volatility Monitoring", fontsize=13, fontweight="bold")
ax2.set_ylabel("Basis Points (bps)")
ax2.set_xlabel("Time (UTC)")
ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()
print("Statistical Anomaly Surveillance Walkthrough Complete.")
