# tests/label_distribution_demo.py
# --------------------------------
"""
Print how many bars get label 0 / 1 / 2 for several ATR multipliers.
Run:  python tests/label_distribution_demo.py
"""

from pathlib import Path
import pandas as pd

# ---- import your own helpers -------------------------------------------
from services.data_utils import load_nt8_ticks, ticks_to_bars
# ------------------------------------------------------------------------

RAW_FILE = Path(__file__).parent.parent / "data" / "nq_3feb_25mar_last.txt"
TIMEFRAME_MIN = 5            # test on 5‑minute aggregation
ATR_MULTS = [1.0, 0.8, 0.6, 0.5]

def main():
    print("Loading raw ticks …")
    ticks_df = load_nt8_ticks(RAW_FILE)

    for mult in ATR_MULTS:
        bars = ticks_to_bars(
            ticks_df,
            timeframe_min=TIMEFRAME_MIN,
            atr_mult=mult          # <‑‑ test different thresholds
        )

        # counts in absolute numbers and in %
        abs_counts = bars["label3"].value_counts().to_dict()
        pct_counts = (
            bars["label3"].value_counts(normalize=True)
            .round(3)
            .mul(100)
            .to_dict()
        )

        print(f"\nATR mult {mult}")
        print("  absolute :", abs_counts)
        print("  percent  :", pct_counts)

if __name__ == "__main__":
    main()
