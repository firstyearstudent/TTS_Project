import pandas as pd, numpy as np
from pathlib import Path

p = Path("results/ber_summary.csv")
if not p.exists():
    raise SystemExit("Chưa có results/ber_summary.csv. Hãy chạy tools/plot_report.py trước.")
df = pd.read_csv(p)

def get(curve, target):
    r = df[(df["Curve"]==curve) & (df["Target BER"]==target)]["Eb/N0 (dB)"]
    return None if r.empty or pd.isna(r.iloc[0]) else float(r.iloc[0])

pairs = [
    ("SISO OFDM (FULL)", "SISO AWGN (FULL)", 1e-3, "OFDM lệch so AWGN"),
    ("2x2 MIMO-OFDM LMMSE", "2x2 MIMO-OFDM ZF", 1e-3, "LMMSE so với ZF"),
]
for a,b,tgt,desc in pairs:
    xa, xb = get(a, tgt), get(b, tgt)
    if xa is None or xb is None:
        print(f"{desc} @ BER={tgt:g}: không nội suy được (thiếu dải).")
    else:
        print(f"{desc} @ BER={tgt:g}: {xa-xb:+.2f} dB ( {a}: {xa:.2f} dB, {b}: {xb:.2f} dB )")
