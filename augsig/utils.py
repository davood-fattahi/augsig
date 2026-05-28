import numpy as np
from scipy.signal import butter, filtfilt


def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)
    return x_norm


def freqfilt(signal, fl_norm, fh_norm, order=4):
    """
    Apply Butterworth filter to a signal.

    Parameters:
        signal   : array-like, input signal.
        fl_norm  : float, normalized low cutoff frequency (0.0 to 1.0, where 1.0 is Nyquist).
        fh_norm  : float, normalized high cutoff frequency (0.0 to 1.0, where 1.0 is Nyquist).
        order    : int, order of the filter (default=4).

    Returns:
        Filtered signal (same shape as input).
    """
    if not (0 <= fl_norm <= 1) or not (0 <= fh_norm <= 1):
        raise ValueError("fl_norm and fh_norm must be between 0 and 1.")
    if fl_norm > fh_norm:
        raise ValueError("fl_norm must be <= fh_norm.")

    if (fl_norm == 0) and (fh_norm == 1):
        return signal
    elif fl_norm == 0:
        b, a = butter(order, fh_norm, btype='lowpass')
        return filtfilt(b, a, signal)
    elif fh_norm == 1:
        b, a = butter(order, fl_norm, btype='highpass')
        return filtfilt(b, a, signal)
    else:
        b, a = butter(order, [fl_norm, fh_norm], btype='band')
        return filtfilt(b, a, signal)
