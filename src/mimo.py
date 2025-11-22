import numpy as np
from tqdm import tqdm
from .qam import map_bits_to_syms, demap_syms_to_bits
from .ofdm import allocate_grid
from .awgn import ebn0_to_sigma2, awgn

def _gen_rayleigh(nr, nt, nsub, rng):
    H = (rng.standard_normal((nsub, nr, nt)) + 1j*rng.standard_normal((nsub, nr, nt))) / np.sqrt(2.0)
    return H.astype(np.complex128)

def mimo_ofdm_ber(M=64, Nt=2, Nr=2, Nfft=512, Ncp=64, data_ratio=0.8125,
                  Eq="ZF", EbN0_dB_list=range(0,31,4), target_err=200, seed=2025):
    assert Nt == 2 and Nr == 2, "Mẫu này fix 2x2 để đơn giản"
    rng = np.random.default_rng(seed)
    k = int(np.log2(M))

    # Số subcarrier dữ liệu
    data_idx, _, _ = allocate_grid(Nfft, data_ratio)
    Ndata = len(data_idx)

    eta_data = Ndata / Nfft
    eta_cp   = 1.0 / (1.0 + Ncp / Nfft)

    ber = []
    for EbN0_dB in tqdm(EbN0_dB_list, desc="MIMO", dynamic_ncols=True):
        sigma2 = ebn0_to_sigma2(EbN0_dB, k, eta_data, eta_cp)
        err = 0; tot = 0

        # batch lớn hơn khi SNR cao để hội tụ nhanh
        Nsym = 8 if EbN0_dB < 14 else (32 if EbN0_dB < 22 else 128)

        while err < target_err:
            # 1) Kênh block-fading theo subcarrier
            H = _gen_rayleigh(Nr, Nt, Ndata, rng)  # (Ndata, Nr, Nt)

            # 2) LS pilot trực giao (2 OFDM symbol)
            p = 1.0 + 0j
            Hhat = np.empty_like(H)
            Hhat[:, :, 0] = (H[:, :, 0]*p + awgn((Ndata, Nr), sigma2)) / p
            Hhat[:, :, 1] = (H[:, :, 1]*p + awgn((Ndata, Nr), sigma2)) / p

            # 3) Dữ liệu
            bits = rng.integers(0, 2, size=Nsym * Ndata * Nt * k, dtype=np.uint8)
            syms = map_bits_to_syms(bits, M).reshape(Nsym, Ndata, Nt)  # (S, I, T)

            # 4) Truyền qua kênh theo từng subcarrier:
            #    y[s,i,:] = H[i,:,:] @ syms[s,i,:]
            Y = np.einsum('irt,sit->sir', H, syms) + awgn((Nsym, Ndata, Nr), sigma2)  # (S, I, R)

            # 5) Equalization (ZF/LMMSE) dùng Ĥ
            alpha = 0.0 if Eq.upper() == "ZF" else sigma2
            xhat = np.empty((Nsym, Ndata, Nt), dtype=np.complex128)
            I = np.eye(Nt, dtype=np.complex128)
            Hh = np.conjugate(Hhat).transpose(0, 2, 1)  # (I, T, R)

            for i in range(Ndata):
                A = Hh[i] @ Hhat[i] + (alpha * I)   # (T,T)
                Ainv = np.linalg.inv(A)
                Hy = np.einsum("ij,sj->si", Hh[i], Y[:, i, :])  # (S,T)
                xhat[:, i, :] = (Ainv @ Hy.T).T

            # 6) Demap & đếm lỗi
            bh = demap_syms_to_bits(xhat.reshape(-1, Nt).ravel(), M)
            e = np.count_nonzero(bits != bh)
            err += e; tot += bits.size

        ber.append(err / tot)

    return np.array(EbN0_dB_list, float), np.array(ber, float)
