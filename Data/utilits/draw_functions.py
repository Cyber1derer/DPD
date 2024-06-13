import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
def plot_psd_complex(*signals, fs: float, N: int = 2048, window: str = 'hann',name: str = 'pic.png'):
    """Compute PSD of multiple signals using Welch method and plot them

    Parameters
    ----------
    signals :
        dictionary of signals to compute PSD,
        format:
        1st variant: {("name_1", ls, lw, marker, markersize): signal_1, ("name_2", ls, lw, marker, markersize): signal_2, ...}
        2nd variant: {"name_1": signal_1, "name_2": signal_2, ...}, format can be choosen separately for each plot
    fs : float              4*122.88
        sampling frequency [MHz]
    N : int, optional
        window size, by default 2048
    window : str, optional
        window type (windows are taken from `sp.signal.windows`), by default 'hann'
    """

    # defaults
    ls = '-'
    lw = 0.3
    marker = '.'
    markersize = 0.5

    plt.figure()
    plt.xlabel('Freq [MHz]')
    plt.ylabel('Power Spectrum [dB]')
    for info, s in signals[0].items():
        if type(info) != tuple:
            info = (info, ls, lw, marker, markersize)

        # default detrend removes dc
        f_psd, signal_psd = sp.signal.welch(x=s, fs=fs, scaling='spectrum', return_onesided=False, detrend=False,
                                            window=sp.signal.get_window(window, N))
        arg_sort = np.argsort(f_psd)
        f_psd = np.roll(f_psd, N // 2, axis=0)
        signal_psd = np.roll(signal_psd, N // 2, axis=0)

        plt.plot(f_psd, 10 * np.log10(signal_psd), label=info[0], ls=info[1], lw=info[2], marker=info[3],
                 markersize=info[4])

    plt.legend()
    plt.grid('both')
    plt.savefig(name)

    plt.show()
def plot_psd_complex2(*signals, fs: float, N: int = 2048, window: str = 'hann',name: str = 'pic.png'):
    """Compute PSD of multiple signals using Welch method and plot them

    Parameters
    ----------
    signals :
        dictionary of signals to compute PSD,
        format:
        1st variant: {("name_1", ls, lw, marker, markersize): signal_1, ("name_2", ls, lw, marker, markersize): signal_2, ...}
        2nd variant: {"name_1": signal_1, "name_2": signal_2, ...}, format can be choosen separately for each plot
    fs : float              4*122.88
        sampling frequency [MHz]
    N : int, optional
        window size, by default 2048
    window : str, optional
        window type (windows are taken from `sp.signal.windows`), by default 'hann'
    """

    # defaults
    ls = '-'
    lw = 0.3
    marker = '.'
    markersize = 0.5

    plt.figure()
    plt.xlabel('Freq [MHz]')
    plt.ylabel('Power Spectrum [dB]')
    for info, s in signals[0].items():
        if type(info) != tuple:
            info = (info, ls, lw, marker, markersize)

        # default detrend removes dc
        f_psd, signal_psd = sp.signal.welch(x=s, fs=fs, scaling='spectrum', return_onesided=False, detrend=False,
                                            window=sp.signal.get_window(window, N))
        arg_sort = np.argsort(f_psd)
        f_psd = np.roll(f_psd, N // 2, axis=0)
        signal_psd = np.roll(signal_psd, N // 2, axis=0)

        plt.plot(f_psd, 10 * np.log10(signal_psd), label=info[0], ls=info[1], lw=info[2], marker=info[3],
                 markersize=info[4])

    plt.legend()
    plt.grid('both')
    plt.savefig(name)

    plt.show()