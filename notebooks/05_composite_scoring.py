# %% [markdown]
# # Notebook 05: Multi-Modal Composite Manipulation Scoring & Regulatory Alerting
# 
# **Project**: Real-Time Crypto Wash-Trade & Market Manipulation Detector  
# **Objective**: Unify counterparty graph topology, EWMA statistical anomalies,
# Benford's Law forensic tests, and order-book spoofing metrics into a single
# Manipulation Probability Index (MPI). Triage alerts and simulate regulatory compliance audit trails.

# %%
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utils import load_config, setup_logger, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.alert_scoring import CompositeAlertEngine

config = load_config()
logger = setup_logger("nb_05_scoring")
print("Composite Manipulation Scoring Engine Loaded.")

# %% [markdown]
# ## 1. Ingest Multi-Modal Market Session
# 
# We simulate a multi-entity trading session containing both genuine market makers and a collusive syndicate.

# %%
stream = generate_synthetic_trade_stream(num_trades=1200, inject_wash_trading=True, random_seed=777)
prep = TradePreprocessor(config=config)
trades_df = prep.normalize_trade_dataframe(stream)

print(f"Loaded {len(trades_df)} cleaned trade ticks.")
print(f"Total Market Notional: ${trades_df['notional_usd'].sum():,.2f} USD")

# %% [markdown]
# ## 2. Execute End-to-End Composite Scoring Engine
# 
# The engine evaluates:
# 1. Graph cycles & Louvain communities ($w_g = 0.35$)
# 2. Volume EWMA & sliding-window Z-score ($w_v = 0.25$)
# 3. Benford First-Digit Kolmogorov-Smirnov test ($w_b = 0.20$)
# 4. Order-cancel ratio & spread spoofing ($w_s = 0.20$)

# %%
engine = CompositeAlertEngine(config=config)
audit_report = engine.evaluate_dataset(trades_df)

print("\n" + "="*60)
print("             MANIPULATION AUDIT REPORT SUMMARY             ")
print("="*60)
print(f"Composite Manipulation Probability Index (MPI): {audit_report['manipulation_probability_index']*100:.2f}%")
print(f"Overall Regulatory Risk Severity:             {audit_report['overall_severity']}")
print(f"Active Surveillance Alerts Triggered:         {audit_report['active_alerts_count']}")
print("="*60)

# %% [markdown]
# ## 3. Sub-Signal Breakdown & Radar Chart
# 
# Visualize how individual signals contribute to the composite manipulation score.

# %%
sub_signals = audit_report["sub_signals"]
categories = list(sub_signals.keys())
values = [sub_signals[k] for k in categories]

# Radar Chart
categories_clean = ["Graph Cycles\n(Wash Rings)", "Volume EWMA\n(Spikes)", "Benford's Law\n(Robotic Sizes)", "Order Spoofing\n(Cancellations)"]
N = len(categories_clean)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
values_plot = values + values[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories_clean, color="black", size=11, fontweight="bold")
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
plt.ylim(0, 1.0)

ax.plot(angles, values_plot, linewidth=2, linestyle="solid", color="#e91e63")
ax.fill(angles, values_plot, color="#e91e63", alpha=0.35)
plt.title(f"Manipulation Signal Vector (MPI: {audit_report['manipulation_probability_index']*100:.1f}% - {audit_report['overall_severity']})", size=14, fontweight="bold", y=1.08)
plt.show()

# %% [markdown]
# ## 4. Regulatory Alert Triage Table
# 
# Convert active alerts into a compliance review dataframe.

# %%
alerts_df = pd.DataFrame(audit_report["alerts"])
if not alerts_df.empty:
    print(f"Displaying top alerts ({len(alerts_df)} total):")
    print(alerts_df[["alert_id", "severity", "type", "score", "description"]].to_string(index=False))
else:
    print("No high-priority alerts triggered during this session.")

# %% [markdown]
# ## 5. Export Compliance Audit Log
# 
# Export the complete findings as JSON for regulatory archiving.

# %%
out_path = os.path.join(project_root, "data", "processed", "compliance_audit_session.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(audit_report, f, indent=2)
print(f"Exported audit log to: {out_path}")
print("Composite scoring walkthrough complete.")
