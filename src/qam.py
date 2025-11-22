import numpy as np

def qam_constellation(M: int) -> np.ndarray:
    m = int(np.sqrt(M)); assert m*m == M
    axis = np.arange(-(m-1), (m-1)+1, 2, dtype=float)
    const = np.array([x + 1j*y for y in axis for x in axis], dtype=np.complex128)
    return const / np.sqrt(np.mean(np.abs(const)**2))  # Es=1

def map_bits_to_syms(bits: np.ndarray, M: int) -> np.ndarray:
    k = int(np.log2(M)); assert bits.size % k == 0
    const = qam_constellation(M)
    b = bits.reshape(-1, k).astype(np.uint8)
    w = (1 << np.arange(k-1, -1, -1, dtype=np.uint32))
    idx = (b * w).sum(axis=1).astype(np.int64)
    return const[idx]

def demap_syms_to_bits(syms: np.ndarray, M: int) -> np.ndarray:
    const = qam_constellation(M); k = int(np.log2(M))
    idx = np.argmin(np.abs(syms[:, None] - const[None, :])**2, axis=1).astype(np.int64)
    return ((idx[:, None] >> np.arange(k-1, -1, -1)) & 1).astype(np.uint8).ravel()
