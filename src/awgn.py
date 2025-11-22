import numpy as np
def ebn0_to_sigma2(EbN0_dB:float,k:int,eta_data:float,eta_cp:float)->float:
    EbN0=10**(EbN0_dB/10); EsN0=EbN0*k*eta_data*eta_cp
    return float(1.0/EsN0)  # sigma^2 (complex)
def awgn(shape, sigma2:float):
    return (np.random.randn(*shape)+1j*np.random.randn(*shape))*np.sqrt(sigma2/2)
