# augsig

**augsig** is a lightweight Python package for augmenting 1D physiological signals such as PPG and ECG. It supports training-time and test-time augmentation using additive noise, smooth geometric warping (Bezier and PCHIP), and artifact-style distortions (burst, powerline, low-frequency noise, linear drift).

---

## Installation

From PyPI:

```bash
pip install augsig
```

Or from source:

```bash
git clone https://github.com/davood-fattahi/augsig.git
cd augsig
pip install .
```

`matplotlib` is required only for the demo script (`tests/test.py`).

---

## Function Reference

These lower-level functions can be called directly for fine-grained control. They are also composed automatically by the augmentation pipeline described in the next section.

### Noise & Artifacts

#### `noisify(signal, snr_db, ...)`

Adds SNR-controlled colored noise to a signal. Returns `(noisy_signal, noise)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signal` | np.ndarray | — | 1D input signal |
| `snr_db` | float | — | Target SNR in dB. `float('inf')` → no noise (returns zeros). `float('-inf')` → unit-power noise (power = 1.0); both outputs are the noise itself — useful when you want to scale by amplitude instead of relative power (see example below) |
| `color` | str | `'white'` | Noise color: `'white'`, `'pink'`, `'brown'`, `'blue'`, `'violet'` |
| `bpass_params` | list | `[0, 1]` | `[fl_norm, fh_norm]` or `[fl_norm, fh_norm, order]`; Nyquist-normalized (0–1) |
| `dist` | str | `'gauss'` | Sample distribution: `'gauss'`, `'uniform'`, `'laplace'`, `'resample'` |
| `resample_pool` | array or `'self'` | `None` | Pool for empirical resampling; required when `dist='resample'` |
| `zero_mean` | bool | `True` | Subtract mean from noise before power scaling |
| `rng` | np.random.Generator | `None` | Seeded RNG for reproducibility |

**Example: SNR-controlled noise**
```python
from augsig.noisifier import noisify

noisy, noise = noisify(signal, snr_db=20, color='pink', bpass_params=[0.01, 0.9], dist='gauss')
```

**Example: Amplitude-controlled noise** — get unit-power noise, then scale manually
```python
_, noise = noisify(signal, snr_db=float('-inf'), color='pink')
noisy = signal + 0.05 * noise
```

**Example: Empirical resampling** — noise samples drawn from the signal itself or an external pool
```python
# Resample from the signal itself (captures its amplitude distribution)
noisy, noise = noisify(signal, snr_db=15, dist='resample', resample_pool='self')

# Resample from an external pool (e.g. a library of real noise recordings)
noise_pool = np.load('real_noise_library.npy')  # 1D array
noisy, noise = noisify(signal, snr_db=15, dist='resample', resample_pool=noise_pool)
```

---

#### `burstify(signal, snr_db, ...)`

Generates localized burst artifacts by masking colored noise into short, randomly placed windows. Returns `(noisy_signal, burst_noise)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signal` | np.ndarray | — | 1D input signal |
| `snr_db` | float | — | Target SNR in dB, applied to the burst noise **after** masking. `float('inf')` → no noise. `float('-inf')` → unit-power burst noise — use this for amplitude-controlled scaling (see example below) |
| `color` | str | `'white'` | Noise color (see `noisify`) |
| `bpass_params` | list | `[0, 0.1]` | Bandpass cutoffs for the base noise |
| `dist` | str | `'laplace'` | Sample distribution (see `noisify`) |
| `resample_pool` | array or `'self'` | `None` | Pool for empirical resampling |
| `n_bursts` | int | `10` | Number of bursts |
| `burst_width` | int or [min, max] | `[1, 10]` | Burst width in samples; a 2-element list samples each width uniformly |
| `burst_base` | float or [min, max] | `0.0` | DC offset added inside each burst |
| `zero_mean` | bool | `False` | Subtract mean from the final burst signal |
| `rng` | np.random.Generator | `None` | Seeded RNG |

**Example: SNR-controlled bursts**
```python
from augsig.noisifier import burstify

noisy, bursts = burstify(signal, snr_db=20, n_bursts=5, burst_width=[5, 20])
```

**Example: Amplitude-controlled bursts** — get unit-power burst noise, then scale manually
```python
_, bursts = burstify(signal, snr_db=float('-inf'), n_bursts=5,
                     burst_width=[5, 20], burst_base=[-0.2, 0.2])
noisy = signal + 0.1 * bursts
```

---

### Warping & Drift

All warping functions share the parameters below.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signal` | np.ndarray | — | 1D input signal |
| `k` | int | `4` | Number of control points / knots. Scale with signal length for consistent warp density |
| `variance` | float | `0.05` | Perturbation strength. Capped internally at `1 / (2k − 4)` to ensure monotonicity |
| `rng` | np.random.Generator | `None` | Seeded RNG |

---

#### `twarp_bezier(signal, k, variance, rng)` / `twarp_pchip(signal, k, variance, rng)`

Nonlinear time-axis warping — the signal is resampled along a smooth monotone mapping. Bezier produces globally smooth distortions; PCHIP produces more localized, segment-wise ones. Returns a warped signal of the same length.

**Example:**
```python
from augsig.warper import twarp_bezier, twarp_pchip

warped = twarp_bezier(signal, k=6, variance=0.02)
warped = twarp_pchip(signal, k=6, variance=0.02)
```

---

#### `adrift_bezier(signal, k, variance, rng)` / `adrift_pchip(signal, k, variance, rng)`

Additive amplitude drift. A smooth curve centered near zero is superimposed on the signal, simulating baseline wander. Returns `signal + drift`.

**Example:**
```python
from augsig.warper import adrift_bezier, adrift_pchip

drifted = adrift_bezier(signal, k=4, variance=0.1)
drifted = adrift_pchip(signal, k=4, variance=0.1)
```

---

#### `amod_bezier(signal, k, variance, rng)` / `amod_pchip(signal, k, variance, rng)`

Multiplicative amplitude modulation. A smooth envelope fluctuating around unity is multiplied into the signal, simulating respiration-induced or pressure-induced amplitude variation. Returns `signal × envelope`.

**Example:**
```python
from augsig.warper import amod_bezier, amod_pchip

modulated = amod_bezier(signal, k=4, variance=0.1)
modulated = amod_pchip(signal, k=4, variance=0.1)
```

---

#### `drift_linear(signal, a, b, rng)`

Adds a linear baseline drift `y = a·t + b`. Slope and intercept can each be a fixed scalar or a `[min, max]` range sampled uniformly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signal` | np.ndarray | — | 1D input signal |
| `a` | float or [min, max] | — | Slope or slope range |
| `b` | float or [min, max] | — | Intercept or intercept range |
| `rng` | np.random.Generator | `None` | Seeded RNG |

**Example: random slope and intercept**
```python
from augsig.warper import drift_linear

drifted = drift_linear(signal, a=[-0.3, 0.3], b=[-0.1, 0.1])
```

---

## Augmentation Pipeline

The `augment()` function and `Augment` class in `augmenter.py` compose all primitives above through a single configuration dictionary. This is the recommended interface for most use cases.

### Config Dict

Augmentation behavior is specified through a nested dictionary. Each top-level key names one augmentation recipe; recipes are independent and each produces one or more output columns.

**Example: config dict structure**
```python
config = {
    "recipe_name": {
        "num_copies": 2,    # stochastic variants to generate from this recipe (default: 1)
        "Add_noise": True,  # enable noise
        "SNRdb": 20,
        # ... other keys
    },
    "another_recipe": { ... },
}
```

Within each recipe, transforms are applied in a fixed order: additive noise → flip → invert → time warp → amplitude drift → amplitude modulation → low-frequency noise → powerline → burst → linear drift → normalization.

**Full config key reference:**

| Key | Type | Description |
|---|---|---|
| `num_copies` | int | Number of variants from this recipe (default: 1) |
| **Noise** | | |
| `Add_noise` | bool | Enable SNR-controlled additive noise |
| `SNRdb` | float | Target SNR in dB |
| `noise_color` | str | `white`, `pink`, `brown`, `violet`, or `blue` |
| `bpass_params` | list | `[fl_norm, fh_norm]` or `[fl_norm, fh_norm, order]`; Nyquist-normalized (0–1) |
| `dist` | str | `gauss`, `uniform`, `laplace`, or `resample` |
| `resample_pool` | array or `"self"` | Required when `dist="resample"` |
| **Geometric** | | |
| `Flip` | bool | Reverse the signal in time |
| `Invert` | bool | Apply `max(signal) − signal` (vertical flip) |
| **Time warping** | | |
| `Bezier_time_warp` | bool | Nonlinear time warp via Bezier curve |
| `Bezier_time_warp_k` | int | Number of control points (default: 4) |
| `Bezier_time_warp_var` | float | Warp strength (default: 0.01) |
| `PCHIP_time_warp` | bool | Nonlinear time warp via PCHIP interpolation |
| `PCHIP_time_warp_k` | int | Number of knots (default: 4) |
| `PCHIP_time_warp_var` | float | Warp strength (default: 0.01) |
| **Amplitude drift** | | |
| `Bezier_amp_drift` | bool | Additive baseline drift via Bezier curve |
| `Bezier_amp_drift_k` | int | Number of control points (default: 4) |
| `Bezier_amp_drift_var` | float | Drift strength (default: 0.05) |
| `PCHIP_amp_drift` | bool | Additive baseline drift via PCHIP |
| `PCHIP_amp_drift_k` | int | Number of knots (default: 4) |
| `PCHIP_amp_drift_var` | float | Drift strength (default: 0.05) |
| **Amplitude modulation** | | |
| `Bezier_amp_mod` | bool | Multiplicative amplitude envelope via Bezier curve |
| `Bezier_amp_mod_k` | int | Number of control points (default: 4) |
| `Bezier_amp_mod_var` | float | Modulation depth (default: 0.05) |
| `PCHIP_amp_mod` | bool | Multiplicative amplitude envelope via PCHIP |
| `PCHIP_amp_mod_k` | int | Number of knots (default: 4) |
| `PCHIP_amp_mod_var` | float | Modulation depth (default: 0.05) |
| **Artifact noise** | | |
| `lf_noise` | bool | Low-frequency additive noise |
| `lf_noise_amplitude` | float | Amplitude scale (default: 0.1) |
| `lf_noise_frequency` | float | Upper cutoff normalized to Nyquist (default: 0.1) |
| `powerline` | bool | Sinusoidal powerline-like interference |
| `powerline_amplitude` | float | Amplitude (default: 0.1) |
| `powerline_frequency` | float | Frequency normalized to Nyquist (default: 0.1) |
| `burst` | bool | Burst artifact noise |
| `burst_amplitude` | float | Burst amplitude scale (default: 0.1) |
| `burst_number` | int | Number of bursts (default: 3) |
| `burst_width` | int or [min, max] | Burst width in samples (default: [1, 10]) |
| `burst_base` | float or [min, max] | DC offset inside each burst (default: 0.0) |
| `burst_frequency` | float | Upper cutoff for burst noise, normalized to Nyquist (default: 0.1) |
| `drift` | bool | Linear baseline drift |
| `drift_a` | float or [min, max] | Slope or slope range (default: [−0.5, 0.5]) |
| `drift_b` | float or [min, max] | Intercept or intercept range (default: [−0.5, 0.5]) |

> **Note:** every generated variant is min-max normalized to [0, 1] by default. Pass `normalize_output=False` to `augment()` or `Augment()` to preserve the original amplitude scale.

---

### Function Style

`augment(data, aug_config, seed=None, normalize_output=True)` returns an `(N, K)` array in one call. The first column is always the unmodified original; each subsequent column is one augmented variant.

**Example:**
```python
import numpy as np
from augsig import augment

signal = np.load("data/sample_ppg.npy")

config = {
    "noise":    {"Add_noise": True, "SNRdb": 20, "noise_color": "white",
                 "bpass_params": [0, 1], "dist": "gauss"},
    "timewarp": {"Bezier_time_warp": True, "Bezier_time_warp_k": 4,
                 "Bezier_time_warp_var": 0.01},
    "drift":    {"drift": True, "drift_a": [-0.3, 0.3], "drift_b": [-0.1, 0.1]},
}

augmented = augment(signal, config, seed=42)               # normalized to [0, 1]
augmented = augment(signal, config, normalize_output=False) # preserve amplitude
print(augmented.shape)  # (N, 4): original + 3 variants
```

---

### Object Style

`Augment(config, seed=None, normalize_output=True)` binds the configuration at construction time, producing a reusable callable — convenient inside training loops or dataset classes.

**Example:**
```python
from augsig import Augment

augmentor = Augment(config, seed=42)
augmented = augmentor(signal)  # equivalent to augment(signal, config, seed=42)
```

---

### Training-Time Augmentation

Omit the seed so each call produces a freshly sampled variant, giving the model a different augmented view across epochs.

**Example:**
```python
import torch
from augsig import Augment

aug_config = {
    "noise": {"Add_noise": True, "SNRdb": 20, "noise_color": "pink",
              "bpass_params": [0, 1], "dist": "gauss"},
    "warp":  {"Bezier_time_warp": True, "Bezier_time_warp_var": 0.01,
              "Bezier_amp_drift": True, "Bezier_amp_drift_var": 0.05},
}

augmentor = Augment(aug_config)  # no seed → different view each epoch

class PPGDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        x = self.data[idx]         # (N,)
        variants = augmentor(x)    # (N, K)
        return variants[:, 1]      # one augmented variant per step
```

---

### Test-Time Augmentation (TTA)

Fix the seed so the augmentation bundle is identical across evaluation runs, then average model predictions over all variants to reduce prediction variance.

**Example:**
```python
import numpy as np
from augsig import augment

tta_config = {
    "flip":  {"Flip": True},
    "noise": {"Add_noise": True, "SNRdb": 20, "noise_color": "white",
              "bpass_params": [0, 1], "dist": "gauss", "num_copies": 3},
    "warp":  {"Bezier_time_warp": True, "Bezier_time_warp_var": 0.01},
}

variants = augment(signal, tta_config, seed=42)  # (N, K), deterministic
preds = np.stack([model(variants[:, k]) for k in range(variants.shape[1])])
ensemble_pred = preds.mean(axis=0)
```

---

## Package Structure

```
augsig/
├── augsig/
│   ├── __init__.py       # Re-exports Augment and augment
│   ├── augmenter.py      # Augment class and augment() entry point
│   ├── noisifier.py      # Additive noise, burst artifacts (noisify, burstify)
│   ├── warper.py         # Bezier and PCHIP warping, linear drift
│   └── utils.py          # Normalization and Butterworth filtering
├── tests/
│   └── test.py           # Informal demo and visualization script
├── data/
│   └── sample_ppg.npy    # Sample PPG signal for testing
└── pyproject.toml        # Packaging config (setuptools, Python >= 3.8)
```

---

## License

MIT License

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
