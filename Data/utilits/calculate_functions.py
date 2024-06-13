import torch
import numpy as np
import scipy


def ACPR_calc(x, fs, mf, afl, afr, mb, amb, n=1024):
    x = x.flatten()
    f, p = scipy.signal.welch(x, fs=fs, nperseg=n, nfft=n, detrend=False, return_onesided=False)
    f = np.roll(f, n // 2, axis=0)
    p = 10 * np.log10(np.roll(p, n // 2, axis=0))
    main_ch_p = np.mean(p[(f >= mf - mb / 2) & (f <= mf + mb / 2)])
    adj_l_ch_p = np.mean(p[(f >= mf + afl - amb / 2) & (f <= mf + afl + amb / 2)])
    adj_r_ch_p = np.mean(p[(f >= mf + afr - amb / 2) & (f <= mf + afr + amb / 2)])
    return [adj_l_ch_p - main_ch_p, adj_r_ch_p - main_ch_p]


def MSE(output, target):
    e = output - target
    return .5 * torch.einsum('ijk,ijk->i', e, torch.conj(e)).mean() / output.shape[2]


def NMSE(X, E):
    return 10 * torch.log10(
        torch.einsum('ijk,ijk->i', E, torch.conj(E)).sum() / torch.einsum('ijk,ijk->i', X, torch.conj(X)).sum())
