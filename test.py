import numpy as np
import matplotlib.pyplot as plt
from .warper import awarp_pchip, twarp_pchip, awarp_bezier, twarp_bezier
from .augmenter import augment as aug



X=np.load("../data/test/signal.npy")
fs = 40
x=X[1,:]
x=x.squeeze()



t = np.arange(len(x)) / fs  # Make sure you set the correct fs
x_pchip = awarp_pchip(x, k=4, variance=0.05)
x_pchip = twarp_pchip(x_pchip, k=4, variance=0.01)

# Plot both signals
plt.figure(figsize=(10, 4))
plt.plot(t, x, label='Original PPG')
plt.plot(t, x_pchip, label='PCHIP Amp & Time Warped PPG')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Augmentation effect")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


x_bezier = awarp_bezier(x, variance=0.05)
x_bezier = twarp_bezier(x_bezier, variance=0.01)

# Plot both signals
plt.figure(figsize=(10, 4))
plt.plot(t, x, label='Original PPG')
plt.plot(t, x_bezier, label='Bezier Amp & Time Warped PPG')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Augmentation effect")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


tta_aug_configs =    {
        "aug1": {"Flip": True},
        "aug2": {"Invert": True},
        "aug3": {"Bezier_time_warp": True, "Bezier_time_var": 0.001},
        "aug4": {"Bezier_amp_warp": True, "Bezier_amp_var": 0.005},
        "aug5": {"PCHIP_amp_warp": True, "PCHIP_amp_var": 0.005},
        "aug6": {"PCHIP_time_warp": True, "PCHIP_time_var": 0.001},
        "aug7": {"Add_noise": True, "SNRdb": 20}
    }

x_aug=aug(x, tta_aug_configs)


# Plot both signals

for xx in x_aug.T:
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label='Original PPG')
    plt.plot(t, xx, label='Bezier Amp & Time Warped PPG')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Augmentation effect")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
