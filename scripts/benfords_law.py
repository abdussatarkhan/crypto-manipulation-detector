"""
Benford's Law Forensic Analysis for Crypto Trade Size Distribution
===================================================================
Applies Newcomb-Benford First-Digit distribution tests to trade volumes and sizes.
Calculates Kolmogorov-Smirnov (KS) D-statistic, Chi-Square goodness-of-fit,
and Mean Absolute Deviation (MAD) to detect robotic, automated wash trading
and human price manipulation that violate natural logarithmic distributions.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import setup_logger, load_config, get_project_root, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor


class BenfordsLawAnalyzer:
    """Forensic statistical validator using Benford's Law."""

    # Theoretical Benford's First-Digit Probabilities: P(d) = log10(1 + 1/d)
    BENFORD_PROBABILITIES = {
        d: np.log10(1.0 + 1.0 / d) for d in range(1, 10)
    }

    # Nigrini (2012) First-Digit MAD Conformity Thresholds
    MAD_THRESHOLDS = {
        "Close Conformity": 0.006,
        "Acceptable Conformity": 0.012,
        "Marginally Acceptable": 0.015,
        "Non-Conforming (Manipulated)": float("inf")
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        self.config = config or load_config()
        self.logger = logger or setup_logger("benfords_analyzer")
        self.benford_cfg = self.config.get("benfords_law", {})
        self.min_sample_size = self.benford_cfg.get("min_sample_size", 100)
        self.ks_alpha = self.benford_cfg.get("ks_alpha", 0.05)

    @staticmethod
    def extract_first_digit(number: float) -> Optional[int]:
        """
        Extracts the leading non-zero digit of any positive floating point number.
        e.g., 0.00452 -> 4, 783.12 -> 7, 0.0 -> None.
        """
        if pd.isna(number) or number <= 0:
            return None
        # Normalize via scientific notation string
        sci_str = f"{float(number):.10e}"
        first_char = sci_str[0]
        if first_char.isdigit() and first_char != "0":
            return int(first_char)
        return None

    def analyze_distribution(self, data_series: pd.Series, series_name: str = "Trade Volume") -> Dict[str, Any]:
        """
        Executes comprehensive Benford analysis on numeric series.

        Args:
            data_series: Numeric Pandas series of trade sizes or transaction quantities.
            series_name: Identifier label for reporting.

        Returns:
            Dictionary containing empirical counts, theoretical counts, KS test, Chi-Square,
            MAD, conformity level, and manipulation probability.
        """
        clean_data = pd.to_numeric(data_series, errors="coerce").dropna()
        clean_data = clean_data[clean_data > 0]

        n = len(clean_data)
        if n < self.min_sample_size:
            self.logger.warning(f"Sample size {n} is below minimum threshold ({self.min_sample_size}) for robust Benford testing.")
            return {
                "series_name": series_name,
                "sample_size": n,
                "status": "INSUFFICIENT_DATA",
                "manipulation_probability": 0.0
            }

        # Extract leading digits
        first_digits = [self.extract_first_digit(x) for x in clean_data]
        valid_digits = [d for d in first_digits if d is not None and 1 <= d <= 9]
        total_valid = len(valid_digits)

        # Count frequencies
        digit_counts = pd.Series(valid_digits).value_counts().reindex(range(1, 10), fill_value=0)
        empirical_probs = digit_counts / total_valid
        theoretical_probs = pd.Series(self.BENFORD_PROBABILITIES)
        expected_counts = theoretical_probs * total_valid

        # 1. Mean Absolute Deviation (MAD)
        abs_diffs = (empirical_probs - theoretical_probs).abs()
        mad = float(abs_diffs.mean())

        # Determine conformity label
        conformity = "Non-Conforming (Manipulated)"
        for label, thresh in self.MAD_THRESHOLDS.items():
            if mad <= thresh:
                conformity = label
                break

        # 2. Chi-Square Goodness-of-Fit Test
        # Sum of (Observed - Expected)^2 / Expected
        chi2_stat, chi2_p = stats.chisquare(f_obs=digit_counts, f_exp=expected_counts)

        # 3. Kolmogorov-Smirnov (KS) Test against Benford Cumulative Distribution
        empirical_cdf = empirical_probs.cumsum().values
        theoretical_cdf = theoretical_probs.cumsum().values
        ks_stat = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

        # Critical value approximation for KS: 1.36 / sqrt(N) at alpha = 0.05
        ks_critical_value = 1.36 / np.sqrt(total_valid)
        is_ks_rejected = ks_stat > ks_critical_value

        # 4. Composite Benford Manipulation Risk Score [0.0 - 1.0]
        # Scales MAD and KS rejection into continuous risk probability
        mad_ratio = min(1.0, mad / 0.030)  # 0.030 represents severe deviation
        ks_ratio = min(1.0, ks_stat / (ks_critical_value * 2.0))
        chi_penalty = 1.0 if chi2_p < 0.001 else (1.0 - chi2_p)

        manipulation_score = 0.5 * mad_ratio + 0.3 * ks_ratio + 0.2 * chi_penalty
        manipulation_score = round(float(np.clip(manipulation_score, 0.0, 1.0)), 4)

        # Identify anomalous outlier digits (digits with > 2x expected occurrence)
        suspicious_digits = []
        for d in range(1, 10):
            ratio = empirical_probs[d] / theoretical_probs[d]
            if ratio >= 2.0 or ratio <= 0.35:
                suspicious_digits.append({
                    "digit": d,
                    "empirical_pct": round(empirical_probs[d] * 100, 2),
                    "expected_pct": round(theoretical_probs[d] * 100, 2),
                    "ratio": round(ratio, 2)
                })

        result = {
            "series_name": series_name,
            "sample_size": total_valid,
            "mad": round(mad, 5),
            "conformity": conformity,
            "chi2_stat": round(float(chi2_stat), 3),
            "chi2_p_value": float(chi2_p),
            "ks_stat": round(ks_stat, 5),
            "ks_critical_value": round(ks_critical_value, 5),
            "ks_rejected": bool(is_ks_rejected),
            "manipulation_probability": manipulation_score,
            "suspicious_digits": suspicious_digits,
            "empirical_proportions": {int(k): round(float(v), 4) for k, v in empirical_probs.items()},
            "theoretical_proportions": {int(k): round(float(v), 4) for k, v in theoretical_probs.items()}
        }

        self.logger.info(
            f"Benford Analysis [{series_name}]: MAD={result['mad']} ({conformity}) | "
            f"KS-stat={result['ks_stat']} (Crit={result['ks_critical_value']}) | "
            f"Risk Score={manipulation_score:.2f}"
        )
        return result

    def analyze_by_entity(self, trades_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Runs per-wallet/per-entity Benford analysis to identify specific wash-trading accounts.
        """
        results = []
        grouped = trades_df.groupby("buyer_address")
        for addr, group in grouped:
            if len(group) >= self.min_sample_size:
                res = self.analyze_distribution(group["quantity"], series_name=f"Wallet {addr[:10]}...")
                res["address"] = addr
                res["entity"] = group["buyer_entity"].iloc[0] if "buyer_entity" in group.columns else "Unknown"
                res["trade_count"] = len(group)
                res["total_volume_usd"] = group["notional_usd"].sum() if "notional_usd" in group.columns else 0.0
                results.append(res)

        if not results:
            return pd.DataFrame()

        df_entities = pd.DataFrame(results)
        df_entities.sort_values("manipulation_probability", ascending=False, inplace=True)
        return df_entities.head(top_n)


def main():
    parser = argparse.ArgumentParser(description="Forensic Benford's Law analysis on crypto trades.")
    parser.add_argument("--input", type=str, default=None, help="Path to trade dataset")
    parser.add_argument("--column", type=str, default="quantity", help="Column to analyze")
    args = parser.parse_args()

    analyzer = BenfordsLawAnalyzer()

    if args.input and os.path.exists(args.input):
        df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    else:
        analyzer.logger.info("Generating synthetic trade stream with non-conforming wash-trading lots...")
        stream = generate_synthetic_trade_stream(num_trades=1000, inject_wash_trading=True)
        prep = TradePreprocessor()
        df = prep.normalize_trade_dataframe(stream)

    res = analyzer.analyze_distribution(df[args.column], series_name=f"{args.column.upper()} Distribution")

    print("\n========================================================")
    print("           BENFORD'S LAW FORENSIC REPORT               ")
    print("========================================================")
    print(f"Sample Size:         {res['sample_size']} trades")
    print(f"Mean Absolute Dev:   {res['mad']:.5f}")
    print(f"Conformity Status:   {res['conformity']}")
    print(f"KS Statistic:        {res['ks_stat']:.5f} (Critical: {res['ks_critical_value']:.5f})")
    print(f"KS Hypothesis Null:  {'REJECTED (ANOMALOUS)' if res['ks_rejected'] else 'ACCEPTED (NATURAL)'}")
    print(f"Chi-Square P-Value:  {res['chi2_p_value']:.4e}")
    print(f"Manipulation Risk:   {res['manipulation_probability']*100:.1f}%")
    print("--------------------------------------------------------")
    print("Digit Distribution Breakdown:")
    print(f"{'Digit':<6}{'Observed %':<15}{'Expected %':<15}{'Delta %':<10}")
    for d in range(1, 10):
        obs = res["empirical_proportions"][d] * 100
        exp = res["theoretical_proportions"][d] * 100
        delta = obs - exp
        print(f"{d:<6}{obs:<15.2f}{exp:<15.2f}{delta:+<10.2f}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
