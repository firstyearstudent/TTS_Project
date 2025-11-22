import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Danh sách ứng viên; script sẽ chỉ vẽ những file thực sự tồn tại
candidates = [
    ("results/awgn_64qam.csv",            "SISO AWGN (FULL)"),
    ("results/awgn_fast_64qam.csv",       "SISO AWGN (FAST)"),
    ("results/ofdm_64qam.csv",            "SISO OFDM (FULL)"),
    ("results/ofdm_fast_64qam.csv",       "SISO OFDM (FAST)"),
    ("results/mimo_zf_64qam.csv",         "2x2 MIMO-OFDM ZF (FULL)"),
    ("results/mimo_zf_fast_64qam.csv",    "2x2 MIMO-OFDM ZF (FAST)"),
    ("results/mimo_lmmse_64qam.csv",      "2x2 MIMO-OFDM LMMSE (FULL)"),
    ("results/mimo_lmmse_fast_64qam.csv", "2x2 MIMO-OFDM LMMSE (FAST)"),
]

pairs = [(Path(p), label) for p, label in candidates if Path(p).exists()]
if not pairs:
    raise SystemExit("Không tìm thấy CSV nào trong 'results/'. Hãy chạy mô phỏng trước.")

plt.figure()
for p, label in pairs:
    df = pd.read_csv(p)
    if "EbN0_dB" not in df or "BER" not in df:
        print(f"File {p} thiếu cột EbN0_dB/BER, bỏ qua.")
        continue
    plt.semilogy(df["EbN0_dB"], df["BER"], "-o", label=label)

plt.grid(True, which="both")
plt.xlabel("E_b/N_0 (dB)")
plt.ylabel("BER")
plt.legend()
plt.tight_layout()
Path("figs").mkdir(exist_ok=True)
out = Path("figs/compare_ber.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
