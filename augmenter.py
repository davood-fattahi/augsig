from utils import normalize_per_instance
from warper import twarp_bezier, awarp_bezier, twarp_pchip, awarp_pchip
import numpy as np


class Augment:
    def __init__(self, config, dtype):
        self.config = config
        self.dtype = dtype

    def __call__(self, signal):
        if self.dtype == "torch":
            augdata=augment_torch(signal, self.config)
        elif self.dtype == "numpy":
            augdata=augment_np(signal, self.config)
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
        augmented = data.copy()

        # Gaussian noise
        if config.get("Add_noise", False):
            snr_db = config["SNRdb"]
            signal_power = np.mean(data ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.random.randn(*data.shape) * np.sqrt(noise_power)
            augmented += noise

        # Flip
        if config.get("Flip", False):
            augmented = np.flip(augmented)

        # Invert
        if config.get("Invert", False):
            augmented = 1.0 - augmented

        # Time/Amplitude Warping
        if any(config.get(k, False) for k in ["Bezier_time_warp", "Bezier_amp_warp", "PCHIP_time_warp", "PCHIP_amp_warp"]):
            signal_np = augmented.copy()
            if config.get("Bezier_time_warp", False):
                signal_np = twarp_bezier(signal_np, variance=config.get("Bezier_time_warp_var", 0.01))
            if config.get("Bezier_amp_warp", False):
                signal_np = awarp_bezier(signal_np, variance=config.get("Bezier_amp_warp_var", 0.05))
            if config.get("PCHIP_time_warp", False):
                signal_np = twarp_pchip(signal_np, variance=config.get("PCHIP_time_warp_var", 0.01))
            if config.get("PCHIP_amp_warp", False):
                signal_np = awarp_pchip(signal_np, variance=config.get("PCHIP_amp_warp_var", 0.05))
            augmented = signal_np

        # Normalize
        augmented = normalize_per_instance(augmented)

        # Append
        aug_versions.append(augmented)

    # Stack into (N, K)
    augdata = np.stack(aug_versions, axis=-1)
    return augdata
