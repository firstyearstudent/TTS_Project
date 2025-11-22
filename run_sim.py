# -*- coding: utf-8 -*-
import argparse, yaml, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from src.eval import siso_awgn_ber, siso_ofdm_ber

def run(cfg: dict, tag: str):
    Path("results").mkdir(exist_ok=True); Path("figs").mkdir(exist_ok=True)

    # read optional limit for bits per SNR point
    max_bits = cfg.get("max_bits_per_point")

    # --- MIMO branch (only if Nt/Nr provided) ---
    if "Nt" in cfg and "Nr" in cfg:
        from src.mimo import mimo_ofdm_ber  # imported lazily to avoid hard dep
        xs, ys = mimo_ofdm_ber(
            M=cfg.get("M",64), Nt=cfg.get("Nt",2), Nr=cfg.get("Nr",2),
            Nfft=cfg.get("Nfft",512), Ncp=cfg.get("Ncp",64),
            data_ratio=cfg.get("data_ratio",0.8125),
            Eq=cfg.get("eq","ZF"),
            EbN0_dB_list=cfg["EbN0_dB"],
            target_err=cfg.get("target_bit_errors",200),
            seed=cfg.get("seed",2025)
        )
        pd.DataFrame({"EbN0_dB": xs, "BER": ys}).to_csv(f"results/{tag}_64qam.csv", index=False)
        plt.figure(); plt.semilogy(xs, ys, "-^", label=f"2x2 MIMO-OFDM {cfg.get('eq','ZF').upper()} ({tag})")
        plt.grid(True, which="both"); plt.xlabel("E_b/N_0 (dB)"); plt.ylabel("BER"); plt.legend()
        plt.savefig(f"figs/{tag}_64qam.png", dpi=150); plt.close()
        print(f"Saved results/{tag}_64qam.csv & figs/{tag}_64qam.png")
        return

    # --- SISO AWGN ---
    if "modulation" in cfg:
        xs, ys = siso_awgn_ber(
            M=cfg.get("M",64),
            EbN0_dB_list=cfg["EbN0_dB"],
            target_err=cfg.get("target_bit_errors",200),
            seed=cfg.get("seed",2025),
            max_bits_per_point=max_bits,
        )
        pd.DataFrame({"EbN0_dB": xs, "BER": ys}).to_csv(f"results/{tag}_64qam.csv", index=False)
        plt.figure(); plt.semilogy(xs, ys, "-o", label=f"SISO-AWGN 64-QAM ({tag})")
        plt.grid(True, which="both"); plt.xlabel("E_b/N_0 (dB)"); plt.ylabel("BER"); plt.legend()
        plt.savefig(f"figs/{tag}_64qam.png", dpi=150); plt.close()
        print(f"Saved results/{tag}_64qam.csv & figs/{tag}_64qam.png")

    # --- SISO OFDM ---
    else:
        xs, ys = siso_ofdm_ber(
            M=cfg.get("M",64),
            Nfft=cfg.get("Nfft",512),
            Ncp=cfg.get("Ncp",64),
            data_ratio=cfg.get("data_ratio",0.8125),
            EbN0_dB_list=cfg["EbN0_dB"],
            target_err=cfg.get("target_bit_errors",200),
            seed=cfg.get("seed",2025),
            max_bits_per_point=max_bits,
        )
        pd.DataFrame({"EbN0_dB": xs, "BER": ys}).to_csv(f"results/{tag}_64qam.csv", index=False)
        plt.figure(); plt.semilogy(xs, ys, "-s", label=f"SISO-OFDM 64-QAM ({tag})")
        plt.grid(True, which="both"); plt.xlabel("E_b/N_0 (dB)"); plt.ylabel("BER"); plt.legend()
        plt.savefig(f"figs/{tag}_64qam.png", dpi=150); plt.close()
        print(f"Saved results/{tag}_64qam.csv & figs/{tag}_64qam.png")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tag = cfg_path.stem
    run(cfg, tag)
