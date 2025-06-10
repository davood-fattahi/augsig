import numpy as np
from scipy.signal import butter, filtfilt

def noisify(signal, snr_db, color='white', bpass_params=None, dist='gauss', resample_pool=None):
    """
    Add colored, SNR-controlled noise to a signal.
    - signal: array-like
    - snr_db: target SNR in dB
    - color: 'white', 'pink', 'brown', 'violet', 'fbounded'
    - bpass_params: (fl_norm, fh_norm, order) for 'fbounded' coloring (bandpass filtering)
    - dist: 'gauss', 'uniform', 'laplace', 'resample'
    - resample_pool: "self" or array for 'resample' dist
    Returns: noisy_signal
    """

    n = len(signal)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    if dist == 'gauss': 
        sigma = np.sqrt(noise_power)    
        noise = sigma * np.random.randn(n)
    elif dist == 'uniform':
        w=np.sqrt(3*noise_power)
        noise = np.random.uniform(low=-w, high=w, size=n)
    elif dist == 'laplace':
        scale = np.sqrt(noise_power/2)
        noise = np.random.laplace(loc=0, scale=scale, size=n)
    elif dist == 'resample':
        if resample_pool is None:
            raise ValueError("resample_pool must be provided for resample distribution")
        elif resample_pool == "self":
            resample_pool = signal
        pool_power = np.mean(resample_pool ** 2)
        scale = np.sqrt(noise_power/pool_power)
        noise = scale * np.random.choice(resample_pool, size=n, replace=True)
    else:
        raise ValueError('Unknown distribution!')

    # --- Color the noise ---
    if color == 'pink':
        noise = color_noise(noise, 1)
    elif color == 'brown':
        noise = color_noise(noise, 2)
    elif color == 'violet':
        noise = color_noise(noise, -2)
    elif color == 'fbounded':
        if bpass_params is None:
            raise ValueError('frequency bounds must be provided for fbounded')
        # bpass_params = [fl_norm, fh_norm, order]
        noise = bandpass(noise, *bpass_params)
    elif color == 'white':
        pass
    else:
        raise ValueError("Unknown color!")
    

    noise = noise - np.mean(noise) # ensure zero mean noise
    return (signal + noise), noise

def color_noise(white, exponent):
    white = white - np.mean(white)  # ensure zero mean
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white))
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid zero
    X_colored = X / (freqs ** (exponent/2))
    colored = np.fft.irfft(X_colored, n=len(white))
    return colored

def bandpass(signal, fl_norm, fh_norm, order=4):
    if fl_norm == 0 & fh_norm ==1: # no filter
        return signal
    else:
        b, a = butter(order, [fl_norm, fh_norm], btype='band')
        return filtfilt(b, a, signal)
