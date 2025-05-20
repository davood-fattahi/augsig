import numpy as np
import neurokit2 as nk
from scipy.signal import welch
import matplotlib.pyplot as plt


def estimate_ppg_simulation_parameters(ppg, sampling_rate=500):
    # Preprocess signal
    signals, info = nk.ppg_process(ppg, sampling_rate=sampling_rate)

    # Heart rate (mean)
    heart_rate = signals["PPG_Rate"].mean()

    # HRV-based IBI randomness (scaled SDNN)
    hrv_metrics = nk.hrv_time(info["PPG_Peaks"], sampling_rate=sampling_rate)
    ibi_randomness = min(hrv_metrics["HRV_SDNN"].values[0] / 1000, 1.0)  # Clamp between 0 and 1

    # Estimate baseline drift (via low-order polynomial detrending)
    baseline = nk.signal_detrend(ppg, method="polynomial", order=2)
    drift = np.std(ppg - baseline)

    # Power spectral density for motion and powerline noise
    freqs, power = welch(ppg, fs=sampling_rate, nperseg=1024)

    # Estimate power in motion range (0.3–1 Hz) and powerline range (49–51 Hz)
    def bandpower(f, p, low, high):
        idx = np.logical_and(f >= low, f <= high)
        return np.trapz(p[idx], f[idx]) if np.any(idx) else 0

    motion_amplitude = bandpower(freqs, power, 0.3, 1.0)
    powerline_amplitude = bandpower(freqs, power, 49, 51)

    # Normalize amplitudes (heuristic scaling to [0, 1] range)
    motion_amplitude = min(motion_amplitude / 1e3, 1.0)
    powerline_amplitude = min(powerline_amplitude / 1e2, 1.0)

    return {
        "heart_rate": heart_rate,
        "ibi_randomness": ibi_randomness,
        "drift": drift,
        "motion_amplitude": motion_amplitude,
        "powerline_amplitude": powerline_amplitude,
    }





def synthesize(ppg_original, sampling_rate, duration):
    params = estimate_ppg_simulation_parameters(ppg_original, sampling_rate=sampling_rate)
    ppg_sim = nk.ppg_simulate(duration=duration, sampling_rate=sampling_rate, **params)
    return ppg_sim



ppg_original=np.load("../datasets/Stanford/X_test.npy")
sampling_rate = 40

ppg_original = ppg_original [1,:]
# Step 2: Estimate simulation parameters from that PPG
params = estimate_ppg_simulation_parameters(ppg_original, sampling_rate=sampling_rate)

# Optional: inspect the estimated parameters
print(params)

# Step 3: Simulate a new signal using the same dynamics
ppg_sim = nk.ppg_simulate(duration=25, sampling_rate=sampling_rate, **params)



t=np.arange(len(ppg_original))/sampling_rate

# Plot both signals
plt.figure(figsize=(10, 4))
plt.plot(t, ppg_original, label='Original PPG')
plt.plot(t, ppg_sim, label='Simulated PPG')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Augmentation effect")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()