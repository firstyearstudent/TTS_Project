import numpy as np
from tqdm import tqdm
from .qam import map_bits_to_syms, demap_syms_to_bits
from .ofdm import ifft_u, fft_u, add_cp, remove_cp, allocate_grid
from .awgn import ebn0_to_sigma2, awgn

def siso_awgn_ber(M=64, EbN0_dB_list=range(0,31,2), target_err=200,
                  seed=2025, max_bits_per_point=None):
    rng = np.random.default_rng(seed)
    k = int(np.log2(M))
    ber = []
    for EbN0_dB in tqdm(EbN0_dB_list, desc="AWGN", dynamic_ncols=True):
        err = 0
        tot = 0
        sigma2 = ebn0_to_sigma2(EbN0_dB, k, 1.0, 1.0)
        n_syms = 10000 if EbN0_dB < 14 else (50000 if EbN0_dB < 22 else 200000)
        while err < target_err and (max_bits_per_point is None or tot < max_bits_per_point):
            bits = rng.integers(0, 2, size=n_syms*k, dtype=np.uint8)
            x = map_bits_to_syms(bits, M)
            y = x + awgn(x.shape, sigma2)
            bh = demap_syms_to_bits(y, M)
            e = np.count_nonzero(bits != bh)
            err += e
            tot += bits.size
        ber.append(err / tot if tot > 0 else float("nan"))
    return np.array(EbN0_dB_list, float), np.array(ber, float)

def siso_ofdm_ber(M=64, Nfft=512, Ncp=64, data_ratio=0.8125,
                  EbN0_dB_list=range(0,31,4), target_err=200,
                  seed=2025, max_bits_per_point=None):
    rng = np.random.default_rng(seed)
    k = int(np.log2(M))
    data_idx, _, _ = allocate_grid(Nfft, data_ratio)
    Ndata = len(data_idx)
    eta_data = Ndata / Nfft
    eta_cp = 1.0 / (1.0 + Ncp / Nfft)
    ber = []
    for EbN0_dB in tqdm(EbN0_dB_list, desc="OFDM", dynamic_ncols=True):
        err = 0
        tot = 0
        sigma2 = ebn0_to_sigma2(EbN0_dB, k, eta_data, eta_cp)
        Nsym = 8 if EbN0_dB < 14 else (32 if EbN0_dB < 22 else 128)
        while err < target_err and (max_bits_per_point is None or tot < max_bits_per_point):
            bits = rng.integers(0, 2, size=Nsym*Ndata*k, dtype=np.uint8)
            syms = map_bits_to_syms(bits, M).reshape(Nsym, Ndata)
            X = np.zeros((Nsym, Nfft), dtype=np.complex128)
            X[:, data_idx] = syms
            xcp = add_cp(ifft_u(X), Ncp)
            ycp = xcp + awgn(xcp.shape, sigma2)
            Y = fft_u(remove_cp(ycp, Ncp))
            y_data = Y[:, data_idx].reshape(-1)
            bh = demap_syms_to_bits(y_data, M)
            e = np.count_nonzero(bits != bh)
            err += e
            tot += bits.size
        ber.append(err / tot if tot > 0 else float("nan"))
    return np.array(EbN0_dB_list, float), np.array(ber, float)
