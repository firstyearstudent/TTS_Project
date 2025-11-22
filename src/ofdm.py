import numpy as np
def ifft_u(X:np.ndarray)->np.ndarray:
    N=X.shape[-1]; return np.fft.ifft(X,axis=-1)*np.sqrt(N)
def fft_u(x:np.ndarray)->np.ndarray:
    N=x.shape[-1]; return np.fft.fft(x,axis=-1)/np.sqrt(N)
def add_cp(x:np.ndarray,Ncp:int)->np.ndarray:
    return np.concatenate([x[..., -Ncp:], x], axis=-1)
def remove_cp(y:np.ndarray,Ncp:int)->np.ndarray:
    return y[..., Ncp:]
def allocate_grid(Nfft:int, data_ratio:float=0.8125):
    dc = Nfft//2 if Nfft%2==0 else 0
    all_idx=np.arange(Nfft); null=set([dc,0,1,Nfft-1])
    usable=np.array([i for i in all_idx if i not in null],dtype=int)
    Ndata=int(np.floor(usable.size*data_ratio))
    data_idx=np.sort(usable[:Ndata])
    null_idx=np.array(sorted(list(set(all_idx)-set(data_idx))),dtype=int)
    pilot_idx=np.array([],dtype=int)
    return data_idx,pilot_idx,null_idx
