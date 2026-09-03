# %% [markdown]
# # Notebook 04: Benford's Law Forensic Analysis on Crypto Trade Sizes
# 
# **Project**: Real-Time Crypto Wash-Trade & Market Manipulation Detector  
# **Objective**: Apply Newcomb-Benford First-Digit distribution tests on cryptocurrency trade sizes.
# Compare empirical distributions against theoretical Benford curves using Kolmogorov-Smirnov (KS) tests,
# Chi-Square goodness-of-fit, and Mean Absolute Deviation (MAD).

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
from scripts.benfords_law import BenfordsLawAnalyzer

config = load_config()
logger = setup_logger("nb_04_benford")
print("Benford's Law Forensic Environment initialized.")

# %% [markdown]
# ## 1. Theoretical Foundation of Benford's Law
# 
# Benford's Law asserts that in naturally occurring multi-scale numeric datasets,
# the probability of the first non-zero digit $d \in \{1, 2, \dots, 9\}$ is:
# $$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$
# 
# Expected proportions:
# - **1**: 30.1%
# - **2**: 17.6%
# - **3**: 12.5%
# - **4**: 9.7%
# - **5**: 7.9%
# - **6**: 6.7%
# - **7**: 5.8%
# - **8**: 5.1%
# - **9**: 4.6%

# %%
analyzer = BenfordsLawAnalyzer(config=config)
print("Theoretical Benford Probabilities:")
for d, p in analyzer.BENFORD_PROBABILITIES.items():
    print(f"Digit {d}: {p*100:.2f}%")

# %% [markdown]
# ## 2. Ingest Trades: Natural Market vs. Manipulated Wash-Trading
# 
# We evaluate two distinct populations:
# 1. Natural trade flow (log-normal, diverse retail/institutional orders)
# 2. Manipulated flow (injected repeating wash trading lots, e.g., 7.77, 9.99, robotic orders)

# %%
# Dataset A: Natural trades
natural_stream = generate_synthetic_trade_stream(num_trades=1000, inject_wash_trading=False, random_seed=42)
prep = TradePreprocessor(config=config)
natural_df = prep.normalize_trade_dataframe(natural_stream)

# Dataset B: Wash-injected trades
manip_stream = generate_synthetic_trade_stream(num_trades=1000, inject_wash_trading=True, random_seed=99)
manip_df = prep.normalize_trade_dataframe(manip_stream)

# %% [markdown]
# ## 3. Forensic Evaluation: Natural vs. Manipulated Flows

# %%
print("--- RUNNING ANALYSIS ON NATURAL STREAM ---")
res_natural = analyzer.analyze_distribution(natural_df["quantity"], series_name="Natural Market Flow")
print(f"MAD: {res_natural['mad']:.5f} | Conformity: {res_natural['conformity']}")
print(f"KS Statistic: {res_natural['ks_stat']:.4f} (Crit: {res_natural['ks_critical_value']:.4f}) | Rejected: {res_natural['ks_rejected']}")
print(f"Manipulation Risk Score: {res_natural['manipulation_probability']*100:.1f}%\n")

print("--- RUNNING ANALYSIS ON MANIPULATED STREAM ---")
res_manip = analyzer.analyze_distribution(manip_df["quantity"], series_name="Wash-Manipulated Flow")
print(f"MAD: {res_manip['mad']:.5f} | Conformity: {res_manip['conformity']}")
print(f"KS Statistic: {res_manip['ks_stat']:.4f} (Crit: {res_manip['ks_critical_value']:.4f}) | Rejected: {res_manip['ks_rejected']}")
print(f"Manipulation Risk Score: {res_manip['manipulation_probability']*100:.1f}%")

# %% [markdown]
# ## 4. Distribution Comparison Visualizations

# %%
digits = list(range(1, 10))
expected_p = [analyzer.BENFORD_PROBABILITIES[d] * 100 for d in digits]
natural_p = [res_natural["empirical_proportions"][d] * 100 for d in digits]
manip_p = [res_manip["empirical_proportions"][d] * 100 for d in digits]

bar_width = 0.28
x = np.arange(len(digits))

plt.figure(figsize=(14, 6))

plt.bar(x - bar_width, natural_p, width=bar_width, label="Natural Market (Empirical)", color="#4caf50", alpha=0.85)
plt.bar(x, manip_p, width=bar_width, label="Wash-Injected Market (Empirical)", color="#f44336", alpha=0.85)
plt.plot(x + bar_width/2, expected_p, marker="o", color="#2196f3", linewidth=2.5, markersize=8, label="Benford Theoretical Curve")

plt.title("Benford's Law First-Digit Distribution: Natural vs. Wash-Trading Market", fontsize=14, fontweight="bold")
plt.xlabel("Leading Digit (1-9)")
plt.ylabel("Observed Frequency (%)")
plt.xticks(x, digits)
plt.legend(loc="upper right", fontsize=11)
plt.show()

# %% [markdown]
# ## 5. Entity-Level Forensic Breakdown
# 
# Drill down into specific wallet addresses to isolate bad actors whose transactions violate Benford's Law.

# %%
entity_report = analyzer.analyze_by_entity(manip_df, top_n=5)
print("\n--- TOP SUSPECTED WALLETS BY BENFORD DEVIATION ---")
if not entity_report.empty:
    print(entity_report[["address", "trade_count", "mad", "conformity", "ks_rejected", "manipulation_probability"]])
else:
    print("No individual wallet reached minimum sample threshold.")

print("\nBenford's Law forensic analysis completed successfully.")
