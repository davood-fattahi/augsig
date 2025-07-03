from augsig.utils import normalize
from augsig.warper import twarp_bezier, adrift_bezier, amod_bezier, twarp_pchip, adrift_pchip, amod_pchip
from augsig.noisifier import noisify
import numpy as np
import neurokit2 as nk


class Augment:
    """
    A callable augmentation wrapper for NumPy 1D signals.

    Args:
        config (dict): augmentation strategy definitions

    Usage:
        augmentor = Augment(config)
        augmented = augmentor(signal)  # signal shape (N,) → output shape (N, K)
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, signal):
        augdata=augment(signal, self.config)
        return augdata



def augment(data: np.ndarray, aug_config: dict) -> np.ndarray:
    """
    Apply multiple augmentation strategies to a 1D NumPy signal.

    Args:
        data (np.ndarray): shape (N,)
        aug_config (dict): nested dictionary defining augmentation versions.

    Returns:
        np.ndarray: shape (N, K), where K = number of augmented versions + 1 (original)
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array (N,)")

    aug_versions = [data.copy()]  # include original

    for config in aug_config.values():

        for m in range(config.get("num_copies", 1)):   # Actually it is not duplicating, but it is regenerating. Due to stochastic factors like noises, the regenerated ones won't be the same.
            augmented = data.copy()

            # Add noise
            if config.get("Add_noise", False):
                snr_db = config["SNRdb"]
                noise_color = config.get("noise_color", 'white')
                bpass_params = config.get("bpass_params", None)
                dist = config.get("dist", 'gauss')
                resample_pool = config.get("resample_pool", None)
                augmented, _ = noisify(augmented, snr_db, color=noise_color, bpass_params=bpass_params, dist=dist, resample_pool=resample_pool)

            # Flip
            if config.get("Flip", False):
                augmented = np.flip(augmented)

            # Invert
            if config.get("Invert", False):
                augmented = 1.0 - augmented

            # Time/Amplitude distortion
            if config.get("Bezier_time_warp", False):
                augmented = twarp_bezier(augmented, k=config.get("Bezier_time_warp_k", 4), variance=config.get("Bezier_time_warp_var", 0.01))
            if config.get("Bezier_amp_drift", False):
                augmented = adrift_bezier(augmented, k=config.get("Bezier_amp_drift_k", 4), variance=config.get("Bezier_amp_drift_var", 0.05))
            if config.get("Bezier_amp_mod", False):
                augmented = amod_bezier(augmented, k=config.get("Bezier_amp_mod_k", 4), variance=config.get("Bezier_amp_mod_var", 0.05))
            if config.get("PCHIP_time_warp", False):
                augmented = twarp_pchip(augmented, k=config.get("PCHIP_time_warp_k", 4), variance=config.get("PCHIP_time_warp_var", 0.01))
            if config.get("PCHIP_amp_drift", False):
                augmented = adrift_pchip(augmented, k=config.get("PCHIP_amp_drift_k", 4), variance=config.get("PCHIP_amp_drift_var", 0.05))
            if config.get("PCHIP_amp_mod", False):
                augmented = amod_pchip(augmented, k=config.get("PCHIP_amp_mod_k", 4), variance=config.get("PCHIP_amp_mod_var", 0.05))


            # Add NeuroKit2 artifacts
            # default sampling frequency = 1000

            # low frequency noise
            if config.get("lf_noise", False):
                augmented = nk.signal_distort(augmented, sampling_rate=config.get("nk_sampling_rate", 1000), noise_amplitude=config.get("lf_noise_amplitude", 0.1), noise_frequency=config.get("nk_sampling_rate", 1000)*config.get("lf_noise_frequency", 0.1))
            # powerline
            if config.get("powerline", False):
                augmented = nk.signal_distort(augmented, sampling_rate=config.get("nk_sampling_rate", 1000), powerline_amplitude=config.get("powerline_amplitude", 0.1), powerline_frequency=config.get("nk_sampling_rate", 1000)*config.get("powerline_frequency", 0.1))
            # burst
            if config.get("burst", False):
                # default sampling frequency = 1000
                augmented = nk.signal_distort(augmented, sampling_rate=config.get("nk_sampling_rate", 1000), artifacts_amplitude=config.get("burst_amplitude", 0.1), artifacts_number = config.get("burst_number", 3), artifacts_frequency=config.get("nk_sampling_rate", 1000)*config.get("burst_frequency", 0.1))
            # drift
            if config.get("drift", False):
                # default sampling frequency = 1000
                augmented = nk.signal_distort(augmented, sampling_rate=config.get("nk_sampling_rate", 1000), linear_drift=True)

            # Normalize
            augmented = normalize(augmented)

            # Append
            aug_versions.append(augmented)

    # Stack into (N, K)
    augdata = np.stack(aug_versions, axis=-1)
    return augdata
