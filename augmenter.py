from utils import normalize_per_instance
from warper import twarp_bezier, awarp_bezier, twarp_pchip, awarp_pchip
import numpy as np

class Augment:
    def __init__(self, config, dtype):
        self.config = config
        self.dtype = dtype

    def __call__(self, signal):
        if dtype == "torch":
            augdata=augment_torch(signal, self.config)
        elif dtype == "numpy":
            augdata=augment_np(signal, self.config)



def augment_torch(data: torch.Tensor, aug_config: dict) -> torch.Tensor:
    """
    Apply multiple augmentation strategies defined in a nested config.

    Args:
        data (torch.Tensor): shape (B, L, P)
        aug_config (dict): nested dictionary. Keys: 'aug1', 'aug2', ..., values: dict of augmentations config
        each of the nested dict of augmentations config may include:
        "SNRdb": float value, final SNR (dB) after adding Gausiian noise,
        "Flip": True/False, vertical fliping, 
        "Invert": True/False, horizontal inverting,
        "Bezier_time_warp": True/False, time warping using Bezier mapping method,
        "Bezier_time_warp_var": max of time warping in Bezier knots, defult 0.01
        "Bezier_amp_warp": True/False, amplitude warping using Bezier mapping method,
        "Bezier_amp_warp_var": max of amplitude deviation in Bezier knots, defult 0.05

        "PCHIP_time_warp": True/False, time warping using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) mapping method,
        "PCHIP_time_warp_var": max of time warping in PCHIP knots, defult 0.01
        "PCHIP_amp_warp": True/False, amplitude warping using PCHIP mapping method,
        "PCHIP_amp_warp_var": max of amplitude deviation in PCHIP knots, defult 0.05

        aug_config is like: 
        aug_config = {
            "aug1": {"SNRdb": 10, "Flip": True, "Bezier_time_warp": True, "Bezier_time_warp_var": 0.01},
            "aug2": {"Invert": True, "Pchip_amp_warp": True},
            ...
        }

    Returns:
        augdata (torch.Tensor): shape (B, L, P, K)
    """
    B, L, P = data.shape
    aug_versions = [data.clone()]  # original version

    for aug_key, config in aug_config.items():
        augmented = data.clone()

        # Gaussian noise
        if config.get("Add_noise", False):
            snr_db = config["SNRdb"]
            signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(data) * torch.sqrt(noise_power)
            augmented = augmented + noise

        # Flip
        if config.get("Flip", False):
            augmented = torch.flip(augmented, dims=[2])

        # Invert
        if config.get("Invert", False):
            augmented = 1.0 - augmented

        # Time/Amplitude Warping
        if any(config.get(k, False) for k in ["Bezier_time_warp", "Bezier_amp_warp", "PCHIP_time_warp", "PCHIP_amp_warp"]):
            warped = torch.zeros_like(augmented)
            for b in range(B):
                signal_np = augmented[b].cpu().numpy().reshape(-1)
                if config.get("Bezier_time_warp", False):
                    signal_np = twarp_bezier(signal_np, variance=config.get("Bezier_time_warp_var", 0.01))
                if config.get("Bezier_amp_warp", False):
                    signal_np = awarp_bezier(signal_np, variance=config.get("Bezier_amp_warp_var", 0.05))
                if config.get("PCHIP_time_warp", False):
                    signal_np = twarp_pchip(signal_np, variance=config.get("PCHIP_time_warp_var", 0.01))
                if config.get("PCHIP_amp_warp", False):
                    signal_np = awarp_pchip(signal_np, variance=config.get("PCHIP_amp_warp_var", 0.05))
                warped[b] = torch.tensor(signal_np, dtype=data.dtype, device=data.device).view(L, P)
            augmented = warped

        # Normalize
        augmented = normalize_per_instance(augmented)

        # Append to list
        aug_versions.append(augmented)

    # Stack into (B, L, P, K)
    augdata = torch.stack(aug_versions, dim=-1)
    return augdata





# Define the NumPy-compatible augment_np function
def augment_np(data: np.ndarray, aug_config: dict) -> np.ndarray:
    """
    Apply multiple augmentation strategies to NumPy signal batches.

    Args:
        data (np.ndarray): shape (B, L, P)
        aug_config (dict): nested dictionary defining augmentation versions.

    Returns:
        np.ndarray: shape (B, L, P, K)
    """
    B, L, P = data.shape
    aug_versions = [data.copy()]  # original version

    for aug_key, config in aug_config.items():
        augmented = data.copy()

        # Gaussian noise
        if config.get("Add_noise", False):
            snr_db = config["SNRdb"]
            signal_power = np.mean(data ** 2, axis=(1, 2), keepdims=True)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.random.randn(*data.shape) * np.sqrt(noise_power)
            augmented += noise

        # Flip
        if config.get("Flip", False):
            augmented = np.flip(augmented, axis=2)

        # Invert
        if config.get("Invert", False):
            augmented = 1.0 - augmented

        # Time/Amplitude Warping
        if any(config.get(k, False) for k in ["Bezier_time_warp", "Bezier_amp_warp", "PCHIP_time_warp", "PCHIP_amp_warp"]):
            warped = np.zeros_like(augmented)
            for b in range(B):
                signal_np = augmented[b].reshape(-1)
                if config.get("Bezier_time_warp", False):
                    signal_np = twarp_bezier(signal_np, variance=config.get("Bezier_time_warp_var", 0.01))
                if config.get("Bezier_amp_warp", False):
                    signal_np = awarp_bezier(signal_np, variance=config.get("Bezier_amp_warp_var", 0.05))
                if config.get("PCHIP_time_warp", False):
                    signal_np = twarp_pchip(signal_np, variance=config.get("PCHIP_time_warp_var", 0.01))
                if config.get("PCHIP_amp_warp", False):
                    signal_np = awarp_pchip(signal_np, variance=config.get("PCHIP_amp_warp_var", 0.05))
                warped[b] = signal_np.reshape(L, P)
            augmented = warped

        # Normalize
        augmented = normalize_per_instance(augmented)

        # Append version
        aug_versions.append(augmented)

    # Stack versions into (B, L, P, K)
    augdata = np.stack(aug_versions, axis=-1)
    return augdata
