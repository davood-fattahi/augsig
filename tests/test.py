import numpy as np
import matplotlib.pyplot as plt

from augsig.utils import normalize
from augsig.warper import (twarp_bezier, twarp_pchip,
                            adrift_bezier, adrift_pchip,
                            amod_bezier, amod_pchip,
                            drift_linear)
from augsig.noisifier import noisify, burstify
from augsig.augmenter import augment, Augment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/sample_ppg.npy"
FS = 40  # Hz — sampling rate of the sample PPG

# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------
try:
    X = np.load(DATA_PATH)
    x = X[0].squeeze()
    print(f"Loaded real data: shape {X.shape}, using row 0 (length {len(x)})\n")
except Exception:
    print("Real data not found — using synthetic PPG-like signal\n")
    t = np.linspace(0, 10, FS * 10)
    x = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)

x = normalize(x)
t = np.arange(len(x)) / FS

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def plot_aug(original, augmented, title, fname):
    plt.figure(figsize=(10, 3))
    plt.plot(t, original, label="Original", color = "blue", linewidth=1.2)
    plt.plot(t, augmented, label=title, color = "red", linewidth=1.0, alpha=0.65)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# ---------------------------------------------------------------------------
# 1. warper — direct function API
#
# Tests each warper function in isolation, bypassing augment().
# Each call gets its own seeded RNG so outputs are independently reproducible.
# ---------------------------------------------------------------------------
print("=== warper ===")

# Time warping: resamples the signal along a non-linear time axis.
# k = number of Bezier/PCHIP control points; variance controls warp strength.
# k can be set relative to signal length; e.g., k=int(len(x) / (2*FS)) places one control point every 2 s.
out = twarp_bezier(x, k=4, variance=0.01, rng=np.random.default_rng(42))
plot_aug(x, out, "Bezier Time Warp", "test_twarp_bezier.png")

out = twarp_pchip(x, k=4, variance=0.01, rng=np.random.default_rng(42))
plot_aug(x, out, "PCHIP Time Warp", "test_twarp_pchip.png")

# Amplitude drift: adds a slow, smooth baseline shift over the signal length.
# k can be set relative to signal length; e.g., k=int(len(x) / (2*FS)) places one control point every 2 s.
out = adrift_bezier(x, k=4, variance=0.2, rng=np.random.default_rng(42))
plot_aug(x, out, "Bezier Amplitude Drift", "test_adrift_bezier.png")

out = adrift_pchip(x, k=4, variance=0.1, rng=np.random.default_rng(42))
plot_aug(x, out, "PCHIP Amplitude Drift", "test_adrift_pchip.png")

# Amplitude modulation: multiplies the signal by a smooth random envelope.
# k can be set relative to signal length; e.g., k=int(len(x) / (2*FS)) places one control point every 2 s.
out = amod_bezier(x, k=4, variance=0.2, rng=np.random.default_rng(42))
plot_aug(x, out, "Bezier Amplitude Modulation", "test_amod_bezier.png")

out = amod_pchip(x, k=4, variance=0.1, rng=np.random.default_rng(42))
plot_aug(x, out, "PCHIP Amplitude Modulation", "test_amod_pchip.png")

# Linear drift: adds a + b*t baseline. Random when a/b are ranges, fixed when scalars.
out = drift_linear(x, a=[-0.3, 0.3], b=[-0.1, 0.1], rng=np.random.default_rng(42))
plot_aug(x, out, "Linear Drift (random)", "test_drift_linear_random.png")

out = drift_linear(x, a=0.2, b=-0.1)  # deterministic — no rng needed
plot_aug(x, out, "Linear Drift (fixed)", "test_drift_linear_fixed.png")

# ---------------------------------------------------------------------------
# 2. noisifier — direct function API
#
# Tests noise injection in isolation. SNR=20 dB is mild noise, chosen to keep
# the augmented signal visually similar to the original.
# ---------------------------------------------------------------------------
print("\n=== noisifier ===")

# Four noise colors: white (flat PSD), pink (1/f), brown (1/f²), violet (f²).
for color in ("white", "pink", "brown", "violet"):
    noisy, _ = noisify(x, snr_db=20, color=color, bpass_params=[0, 1], dist="gauss", rng=np.random.default_rng(42))
    plot_aug(x, noisy, f"Noisify ({color}, SNR=20dB)", f"test_noisify_{color}.png")

# Bandpass-filtered pink noise: retains only 0.05–0.5 (Nyquist-normalized), order=4.
noisy, _ = noisify(x, snr_db=20, color="pink", bpass_params=[0.05, 0.5, 4], dist="gauss", rng=np.random.default_rng(42))
plot_aug(x, noisy, "Pink Noise + Bandpass (SNR=20dB)", "test_noisify_pink_bp.png")

# Resampled noise: noise samples are drawn (with replacement) from a pool signal.
# Here the signal itself is the pool ("self"); it can be any 1D array, e.g. a
# separate recording of known noise or another subject's signal.
noisy, _ = noisify(x, snr_db=20, color="white", bpass_params=[0, 1],
                   dist="resample", resample_pool="self", rng=np.random.default_rng(42))
plot_aug(x, noisy, "Resampled Noise (pool=self, SNR=20dB)", "test_noisify_resample.png")

# Burst noise: sporadic high-amplitude spikes simulating motion artifacts.
burst_sig, _ = burstify(x, snr_db=10, n_bursts=5, burst_width=[int(0.1*FS), int(0.5*FS)], burst_base=0.0, rng=np.random.default_rng(42))
plot_aug(x, burst_sig, "Burst Noise", "test_burstify.png")

# ---------------------------------------------------------------------------
# 3. augmenter — augment()
#
# Tests each augmentation type through the high-level augment() API.
# seed=42 is passed to every call so plots are reproducible across runs.
# Output columns: col 0 = original, col 1 = augmented variant.
# ---------------------------------------------------------------------------
print("\n=== augmenter: augment() ===")

cfg = {"r": {"Add_noise": True, "SNRdb": 20, "noise_color": "white", "bpass_params": [0, 1], "dist": "gauss"}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Add_noise (white, SNR=20dB)", "test_aug_add_noise.png")

cfg = {"r": {"Add_noise": True, "SNRdb": 20, "noise_color": "pink", "bpass_params": [0, 1], "dist": "gauss"}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Add_noise (pink, SNR=20dB)", "test_aug_add_noise_pink.png")

cfg = {"r": {"Flip": True}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Flip", "test_aug_flip.png")

cfg = {"r": {"Invert": True}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Invert", "test_aug_invert.png")

# k can be set relative to signal length; e.g., k=int(len(x) / (2*FS)) places one control point every 2 s.
cfg = {"r": {"Bezier_time_warp": True, "Bezier_time_warp_k": 4, "Bezier_time_warp_var": 0.01}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Bezier_time_warp", "test_aug_bezier_timewarp.png")

cfg = {"r": {"PCHIP_time_warp": True, "PCHIP_time_warp_k": 4, "PCHIP_time_warp_var": 0.01}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "PCHIP_time_warp", "test_aug_pchip_timewarp.png")

# k can be set relative to signal length; e.g., k=int(len(x) / (2*FS)) places one control point every 2 s.
cfg = {"r": {"Bezier_amp_drift": True, "Bezier_amp_drift_k": 4, "Bezier_amp_drift_var": 0.2}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Bezier_amp_drift", "test_aug_bezier_ampdrift.png")

cfg = {"r": {"PCHIP_amp_drift": True, "PCHIP_amp_drift_k": 4, "PCHIP_amp_drift_var": 0.1}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "PCHIP_amp_drift", "test_aug_pchip_ampdrift.png")

# k can be set relative to signal length; e.g., k=int(len(x) / (2*FS)) places one control point every 2 s.
cfg = {"r": {"Bezier_amp_mod": True, "Bezier_amp_mod_k": 4, "Bezier_amp_mod_var": 0.2}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "Bezier_amp_mod", "test_aug_bezier_ampmod.png")

cfg = {"r": {"PCHIP_amp_mod": True, "PCHIP_amp_mod_k": 4, "PCHIP_amp_mod_var": 0.1}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "PCHIP_amp_mod", "test_aug_pchip_ampmod.png")

cfg = {"r": {"lf_noise": True, "lf_noise_amplitude": 0.05, "lf_noise_frequency": 0.05}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "lf_noise", "test_aug_lf_noise.png")

# Powerline: frequency is Nyquist-normalized (0.3 ≈ 6 Hz at FS=40, proxy for 50/60 Hz at higher fs).
# At f_norm=1 (Nyquist), sin(π·idx) = 0 for all integer idx, so amplitude becomes
# phase-dependent — use sub-Nyquist frequencies to avoid that degenerate case.
cfg = {"r": {"powerline": True, "powerline_amplitude": 0.05, "powerline_frequency": 0.3}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "powerline", "test_aug_powerline.png")

cfg = {"r": {"burst": True, "burst_amplitude": 0.2, "burst_number": 5, "burst_width": [5, 20], "burst_frequency": 0.05}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "burst", "test_aug_burst.png")

cfg = {"r": {"drift": True, "drift_a": 0.5, "drift_b": 0.5}}
out = augment(x, cfg, seed=42)
plot_aug(out[:, 0], out[:, 1], "drift (linear)", "test_aug_drift.png")

# ---------------------------------------------------------------------------
# 4. augmenter — Augment class
#
# The Augment class holds config and seed as state, making it convenient for
# repeated calls (e.g. in a DataLoader). With seed set, each __call__ resets
# the RNG to the same state, producing identical output — useful for TTA where
# the same augmentation must be applied consistently.
# ---------------------------------------------------------------------------
print("\n=== augmenter: Augment class ===")

cfg = {"noise": {"Add_noise": True, "SNRdb": 20, "noise_color": "white", "bpass_params": [0, 1], "dist": "gauss"}}
augmentor = Augment(cfg, seed=42)
out = augmentor(x)

# Verify that a seeded Augment produces identical outputs on repeated calls.
augmentor_seeded = Augment(cfg, seed=42)
out_a = augmentor_seeded(x)
out_b = augmentor_seeded(x)
print("Seeded reproducibility:", np.allclose(out_a, out_b))

# ---------------------------------------------------------------------------
# 5. augmenter — num_copies
#
# num_copies generates multiple stochastic variants from the same config key.
# Output shape is (N, 1 + num_copies): col 0 is the original.
# ---------------------------------------------------------------------------
print("\n=== augmenter: num_copies ===")

cfg = {"r": {"num_copies": 3, "Add_noise": True, "SNRdb": 20, "noise_color": "white", "bpass_params": [0, 1], "dist": "gauss"}}
out = augment(x, cfg, seed=42)
print(f"num_copies=3 → output shape: {out.shape}")

# ---------------------------------------------------------------------------
# 6. augmenter — TTA/ITA bundle
#
# Multiple config keys produce one augmented variant each.
# This mirrors a real TTA/ITA setup where diverse augmentations are ensembled
# at inference time to improve prediction robustness.
# ---------------------------------------------------------------------------
print("\n=== augmenter: TTA/ITA bundle ===")

tta_ita_config = {
    "flip":        {"Flip": True},
    "invert":      {"Invert": True},
    "noise":       {"Add_noise": True, "SNRdb": 20, "noise_color": "white", "bpass_params": [0, 1], "dist": "gauss"},
    "bez_twarp":   {"Bezier_time_warp": True, "Bezier_time_warp_k": 4, "Bezier_time_warp_var": 0.01},
    "pchip_twarp": {"PCHIP_time_warp": True, "PCHIP_time_warp_k": 4, "PCHIP_time_warp_var": 0.01},
    "bez_amod":    {"Bezier_amp_mod": True, "Bezier_amp_mod_k": 4, "Bezier_amp_mod_var": 0.05},
    "pchip_amod":  {"PCHIP_amp_mod": True, "PCHIP_amp_mod_k": 4, "PCHIP_amp_mod_var": 0.05},
    "drift":       {"drift": True, "drift_a": [-0.3, 0.3], "drift_b": [-0.1, 0.1]},
}

out = augment(x, tta_ita_config, seed=42)
print(f"TTA/ITA bundle → output shape: {out.shape}")


if __name__ == "__main__":
    pass
