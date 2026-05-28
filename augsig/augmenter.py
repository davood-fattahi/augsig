from augsig.utils import normalize
from augsig.warper import (
    twarp_bezier, adrift_bezier, amod_bezier,
    twarp_pchip, adrift_pchip, amod_pchip,
    drift_linear,
)
from augsig.noisifier import noisify, burstify
import numpy as np


class Augment:
    """
    A callable augmentation wrapper for NumPy 1D signals.

    Args:
        config (dict): augmentation strategy definitions
        seed (int, optional): random seed for reproducibility
        normalize_output (bool): if True, min-max normalize each variant to [0, 1] (default: True)

    Usage:
        augmentor = Augment(config)
        augmented = augmentor(signal)  # signal shape (N,) -> output shape (N, K)
    """

    def __init__(self, config, seed=None, normalize_output=True):
        self.config = config
        self.seed = seed
        self.normalize_output = normalize_output

    def __call__(self, signal, seed=None):
        seed = self.seed if seed is None else seed
        augdata = augment(signal, self.config, seed=seed, normalize_output=self.normalize_output)
        return augdata


def augment(data: np.ndarray, aug_config: dict, seed=None, normalize_output=True) -> np.ndarray:
    """
    Apply multiple augmentation strategies to a 1D NumPy signal.

    Args:
        data (np.ndarray): shape (N,)
        aug_config (dict): nested dictionary defining augmentation versions.
        seed (int, optional): random seed for reproducibility
        normalize_output (bool): if True, min-max normalize each variant to [0, 1] (default: True)

    Returns:
        np.ndarray: shape (N, K), where K = number of augmented versions + 1 (original)
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array (N,)")

    rng = np.random.default_rng(seed)
    aug_versions = [data.copy()]

    for config in aug_config.values():

        # num_copies regenerates the recipe each time; stochastic variants differ
        for m in range(config.get("num_copies", 1)):
            augmented = data.copy()

            # Add noise
            if config.get("Add_noise", False):
                snr_db = config["SNRdb"]
                noise_color = config.get("noise_color", 'white')
                bpass_params = config.get("bpass_params", [0, 1])
                dist = config.get("dist", 'gauss')
                resample_pool = config.get("resample_pool", None)
                augmented, _ = noisify(
                    augmented,
                    snr_db,
                    color=noise_color,
                    bpass_params=bpass_params,
                    dist=dist,
                    resample_pool=resample_pool,
                    rng=rng,
                )

            # Flip
            if config.get("Flip", False):
                augmented = np.flip(augmented)

            # Invert
            if config.get("Invert", False):
                augmented = np.max(augmented) - augmented

            # Time/Amplitude distortion
            if config.get("Bezier_time_warp", False):
                augmented = twarp_bezier(
                    augmented,
                    k=config.get("Bezier_time_warp_k", 4),
                    variance=config.get("Bezier_time_warp_var", 0.01),
                    rng=rng,
                )
            if config.get("Bezier_amp_drift", False):
                augmented = adrift_bezier(
                    augmented,
                    k=config.get("Bezier_amp_drift_k", 4),
                    variance=config.get("Bezier_amp_drift_var", 0.05),
                    rng=rng,
                )
            if config.get("Bezier_amp_mod", False):
                augmented = amod_bezier(
                    augmented,
                    k=config.get("Bezier_amp_mod_k", 4),
                    variance=config.get("Bezier_amp_mod_var", 0.05),
                    rng=rng,
                )
            if config.get("PCHIP_time_warp", False):
                augmented = twarp_pchip(
                    augmented,
                    k=config.get("PCHIP_time_warp_k", 4),
                    variance=config.get("PCHIP_time_warp_var", 0.01),
                    rng=rng,
                )
            if config.get("PCHIP_amp_drift", False):
                augmented = adrift_pchip(
                    augmented,
                    k=config.get("PCHIP_amp_drift_k", 4),
                    variance=config.get("PCHIP_amp_drift_var", 0.05),
                    rng=rng,
                )
            if config.get("PCHIP_amp_mod", False):
                augmented = amod_pchip(
                    augmented,
                    k=config.get("PCHIP_amp_mod_k", 4),
                    variance=config.get("PCHIP_amp_mod_var", 0.05),
                    rng=rng,
                )

            # Low-frequency noise
            if config.get("lf_noise", False):
                _, lf_noise = noisify(
                    signal=augmented,
                    snr_db=float('-inf'),
                    color='white',
                    bpass_params=[0.0, config.get("lf_noise_frequency", 0.1)],
                    dist='gauss',
                    resample_pool=None,
                    zero_mean=True,
                    rng=rng,
                )
                augmented = augmented + config.get("lf_noise_amplitude", 0.1) * lf_noise

            # Powerline interference (Nyquist-normalized frequency)
            if config.get("powerline", False):
                n = len(augmented)
                idx = np.arange(n)
                amp = config.get("powerline_amplitude", 0.1)
                f_norm = config.get("powerline_frequency", 0.1)
                phase = rng.uniform(0, 2 * np.pi)
                powerline_noise = amp * np.sin(np.pi * f_norm * idx + phase)
                augmented = augmented + powerline_noise

            # Burst artifacts
            if config.get("burst", False):
                _, burst_noise = burstify(
                    signal=augmented,
                    snr_db=float('-inf'),
                    color='white',
                    bpass_params=[0.0, config.get("burst_frequency", 0.1)],
                    dist='laplace',
                    resample_pool=None,
                    n_bursts=config.get("burst_number", 3),
                    burst_width=config.get("burst_width", [1, 10]),
                    burst_base=config.get("burst_base", 0.0),
                    zero_mean=False,
                    rng=rng,
                )
                augmented = augmented + config.get("burst_amplitude", 0.1) * burst_noise

            # Linear drift
            if config.get("drift", False):
                augmented = drift_linear(
                    augmented,
                    a=config.get("drift_a", [-0.5, 0.5]),
                    b=config.get("drift_b", [-0.5, 0.5]),
                    rng=rng,
                )

            if normalize_output:
                augmented = normalize(augmented)

            aug_versions.append(augmented)

    return np.stack(aug_versions, axis=-1)
