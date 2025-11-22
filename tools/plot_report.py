import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Tự dò các CSV có sẵn
candidates = [
    ("results/awgn_64qam.csv",            "SISO AWGN (FULL)"),
    ("results/awgn_fast_64qam.csv",       "SISO AWGN (FAST)"),
    ("results/ofdm_64qam.csv",            "SISO OFDM (FULL)"),
    ("results/ofdm_fast_64qam.csv",       "SISO OFDM (FAST)"),
    ("results/mimo_zf_64qam.csv",         "2x2 MIMO-OFDM ZF"),
    ("results/mimo_zf_fast_64qam.csv",    "2x2 MIMO-OFDM ZF (FAST)"),
    ("results/mimo_lmmse_64qam.csv",      "2x2 MIMO-OFDM LMMSE"),
    ("results/mimo_lmmse_fast_64qam.csv", "2x2 MIMO-OFDM LMMSE (FAST)"),
]
pairs = [(Path(p), label) for p, label in candidates if Path(p).exists()]
if not pairs:
    raise SystemExit("Không tìm thấy CSV nào trong 'results/'. Hãy chạy mô phỏng trước.")

def snr_at_ber(xs, ys, target):
    """Nội suy log(BER) để tìm Eb/N0 tại BER=target; trả None nếu không cắt."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ys = np.clip(ys, 1e-15, 1.0)  # tránh log10(0)
    logy = np.log10(ys)
    t = np.log10(target)
    # tìm đoạn [i,i+1] sao cho đường đi qua t
    idx = np.where((logy[:-1]-t)*(logy[1:]-t) <= 0)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    x1, x2 = xs[i], xs[i+1]
    y1, y2 = logy[i], logy[i+1]
    if y2 == y1:
        return None
    return float(x1 + (t - y1) * (x2 - x1) / (y2 - y1))

plt.figure()
rows = []
for p, label in pairs:
    df = pd.read_csv(p)
    x = df["EbN0_dB"].to_numpy()
    y = df["BER"].to_numpy()
    y_plot = np.clip(y, 1e-15, 1.0)
    plt.semilogy(x, y_plot, "-o", label=label)
    for tgt in (1e-3, 1e-4):
        val = snr_at_ber(x, y, tgt)
        rows.append({"Curve": label, "Target BER": tgt, "Eb/N0 (dB)": val})
        if val is not None:
            plt.semilogy([val], [tgt], marker="^")

plt.grid(True, which="both")
plt.xlabel("E_b/N_0 (dB)")
plt.ylabel("BER")
plt.legend()
plt.tight_layout()
Path("figs").mkdir(exist_ok=True)
plt.savefig("figs/compare_ber.png", dpi=150)
print("Saved figs/compare_ber.png")

# Ghi bảng tóm tắt; nếu file bị mở thì lưu file mới kèm timestamp
Path("results").mkdir(exist_ok=True)
out = Path("results/ber_summary.csv")
df_out = pd.DataFrame(rows)
try:
    df_out.to_csv(out, index=False)
    print(f"Saved {out}")
except PermissionError:
    alt = Path("results") / f"ber_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"
    df_out.to_csv(alt, index=False)
    print(f"'{out}' đang bị ứng dụng khác mở; đã lưu tạm vào {alt}")
