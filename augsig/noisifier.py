import numpy as np
from augsig.utils import freqfilt


def burst_mask(noise, n_bursts, burst_width, burst_base, burst_onset):
    """
    Apply burst masking to a noise signal using fully specified burst parameters.

    Args:
        noise (np.ndarray): input noise (1D)
        n_bursts (int): number of bursts
        burst_width (int or 1D array-like of length n_bursts): burst widths in samples
        burst_base (float or 1D array-like of length n_bursts): DC base added to each burst
        burst_onset (int or 1D array-like of length n_bursts): burst onset indices

    Returns:
        np.ndarray: burst-masked noise
    """
    noise = noise.squeeze()
    n = len(noise)

    if np.isscalar(burst_width):
        burst_width = np.full(n_bursts, int(burst_width))
    else:
        burst_width = np.asarray(burst_width, dtype=int)

    if np.isscalar(burst_base):
        burst_base = np.full(n_bursts, float(burst_base))
    else:
        burst_base = np.asarray(burst_base, dtype=float)

    if np.isscalar(burst_onset):
        burst_onset = np.full(n_bursts, int(burst_onset))
    else:
        burst_onset = np.asarray(burst_onset, dtype=int)

    if len(burst_width) != n_bursts:
        raise ValueError("burst_width must be a scalar or a 1D array of length n_bursts")
    if len(burst_base) != n_bursts:
        raise ValueError("burst_base must be a scalar or a 1D array of length n_bursts")
    if len(burst_onset) != n_bursts:
        raise ValueError("burst_onset must be a scalar or a 1D array of length n_bursts")

    burst_noise = np.zeros(n, dtype=float)

    for i in range(n_bursts):
        w = int(burst_width[i])
        b = float(burst_base[i])
        start = int(burst_onset[i])

        if w <= 0:
            continue
        if start < 0:
            start = 0
        if start >= n:
            continue

        end = min(start + w, n)
        burst_noise[start:end] = noise[start:end] + b

    return burst_noise


def noisify(signal, snr_db, color='white', bpass_params=[0, 1], dist='gauss',
            resample_pool=None, zero_mean=True, rng=None):
    """
    Add colored, SNR-controlled noise to a signal.

    Args:
        signal: 1D array-like input signal.
        snr_db: Target SNR in dB.
            - float('inf')  -> no noise; returns (signal, zeros).
            - float('-inf') -> unit-power noise (power=1.0); both outputs are the noise
                               itself (signal is ignored).
        color: Noise color -- 'white', 'pink', 'brown', 'blue', or 'violet'.
        bpass_params: [fl_norm, fh_norm] or [fl_norm, fh_norm, order]; Nyquist-normalized (0-1).
        dist: Sample distribution -- 'gauss', 'uniform', 'laplace', or 'resample'.
        resample_pool: Required when dist='resample'. Pass 'self' to resample from the signal,
                       or supply an external array.
        zero_mean: If True, subtract the mean from noise before power scaling (default: True).
        rng: np.random.Generator for reproducibility.

    Returns:
        (noisy_signal, noise): tuple of two 1D arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    if snr_db == float('inf'):
        return signal, np.zeros_like(signal)

    if snr_db == float('-inf'):
        target_noise_power = 1.0
    elif isinstance(snr_db, (float, int)):
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear
    else:
        raise ValueError(f"Wrong SNR value: {snr_db}")

    n = len(signal)

    if dist == 'gauss':
        noise = rng.standard_normal(n)
    elif dist == 'uniform':
        noise = rng.uniform(low=-1.0, high=1.0, size=n)
    elif dist == 'laplace':
        noise = rng.laplace(loc=0.0, scale=1.0, size=n)
    elif dist == 'resample':
        if resample_pool is None:
            raise ValueError("resample_pool must be provided for resample distribution")
        elif resample_pool == "self":
            resample_pool = signal
        noise = rng.choice(resample_pool, size=n, replace=True)
    else:
        raise ValueError('Unknown distribution!')

    if color == 'pink':
        noise = color_noise(noise, 1)
    elif color == 'brown':
        noise = color_noise(noise, 2)
    elif color == 'violet':
        noise = color_noise(noise, -2)
    elif color == 'blue':
        noise = color_noise(noise, -1)
    elif color == 'white':
        pass
    else:
        raise ValueError("Unknown color!")

    noise = freqfilt(noise, *bpass_params)

    if zero_mean:
        noise = noise - np.mean(noise)

    current_noise_power = np.mean(noise ** 2)
    if current_noise_power <= 0:
        raise ValueError("Filtered noise power became zero; cannot rescale to target SNR.")

    scale = np.sqrt(target_noise_power / (current_noise_power + 1e-12))
    noise = scale * noise

    if snr_db == float('-inf'):
        return noise, noise

    return (signal + noise), noise


def color_noise(white, exponent):
    white = white - np.mean(white)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white))
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    X_colored = X / (freqs ** (exponent / 2))
    colored = np.fft.irfft(X_colored, n=len(white))
    return colored


def burstify(signal, snr_db, color='white', bpass_params=[0, 0.1], dist='laplace',
             resample_pool=None, n_bursts=10, burst_width=[1, 10], burst_base=0.0,
             zero_mean=False, rng=None):
    """
    Generate localized burst artifacts and optionally add them to a signal.

    Args:
        signal: 1D array-like input signal.
        snr_db: Target SNR in dB, applied after masking.
            - float('inf')  -> no burst noise; returns (signal, zeros).
            - float('-inf') -> unit-power burst noise; both outputs are the noise itself.
        color: Noise color -- 'white', 'pink', 'brown', 'blue', or 'violet'.
        bpass_params: [fl_norm, fh_norm] or [fl_norm, fh_norm, order]; Nyquist-normalized (0-1).
        dist: Sample distribution -- 'gauss', 'uniform', 'laplace', or 'resample'.
        resample_pool: Required when dist='resample'.
        n_bursts: Number of burst events (default: 10).
        burst_width: Burst width in samples. Scalar or [min, max] interval.
        burst_base: DC offset added inside each burst. Scalar or [min, max] interval.
        zero_mean: If True, subtract mean from burst noise before power scaling (default: False).
        rng: np.random.Generator for reproducibility.

    Returns:
        (noisy_signal, burst_noise): tuple of two 1D arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    if snr_db == float('inf'):
        return signal, np.zeros_like(signal)

    n = len(signal)

    _, noise = noisify(
        signal=signal,
        snr_db=float('-inf'),
        color=color,
        bpass_params=bpass_params,
        dist=dist,
        resample_pool=resample_pool,
        zero_mean=False,
        rng=rng,
    )

    if n_bursts > 0:
        if (isinstance(burst_width, (list, tuple, np.ndarray))
                and len(burst_width) == 2 and np.isscalar(burst_width[0])):
            burst_width_arr = rng.integers(
                int(burst_width[0]), int(burst_width[1]) + 1, size=n_bursts
            )
        elif np.isscalar(burst_width):
            burst_width_arr = np.full(n_bursts, int(burst_width))
        else:
            raise ValueError("burst_width must be an int or a length-2 interval")

        if (isinstance(burst_base, (list, tuple, np.ndarray))
                and len(burst_base) == 2 and np.isscalar(burst_base[0])):
            burst_base_arr = rng.uniform(
                float(burst_base[0]), float(burst_base[1]), size=n_bursts
            )
        elif np.isscalar(burst_base):
            burst_base_arr = np.full(n_bursts, float(burst_base))
        else:
            raise ValueError("burst_base must be a float/int or a length-2 interval")

        burst_onset_arr = np.array([
            rng.integers(0, max(1, n - int(w) + 1)) if int(w) > 0 else 0
            for w in burst_width_arr
        ], dtype=int)

        noise = burst_mask(
            noise=noise,
            n_bursts=n_bursts,
            burst_width=burst_width_arr,
            burst_base=burst_base_arr,
            burst_onset=burst_onset_arr,
        )

    if zero_mean:
        noise = noise - np.mean(noise)

    if snr_db == float('-inf'):
        target_noise_power = 1.0
    else:
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear

    current_noise_power = np.mean(noise ** 2)
    if current_noise_power > 0:
        scale = np.sqrt(target_noise_power / (current_noise_power + 1e-12))
        noise = scale * noise

    if snr_db == float('-inf'):
        return noise, noise

    return (signal + noise), noise
