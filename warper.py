import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.interpolate import PchipInterpolator


def rand_knots(k=4, variance=0.05):
    x_vals = np.append(np.append(0, np.arange(0, 1, 1/(k-2)) + 1/(2*k - 4)), 1)
    y_vals = x_vals.copy()

    x_vals[1:-1] += np.random.uniform(-variance, variance, size=k-2)
    y_vals[1:-1] += np.random.uniform(-variance, variance, size=k-2)

    x_vals = np.clip(x_vals, 0, 1)
    y_vals = np.clip(y_vals, 0, 1)
    x_vals[1:-1] = np.sort(x_vals[1:-1])
    return x_vals, y_vals

def bezier_curve(t, points):
    P0, P1, P2, P3 = points
    return (
        (1 - t)**3 * P0 +
        3 * (1 - t)**2 * t * P1 +
        3 * (1 - t) * t**2 * P2 +
        t**3 * P3
    )

def twarp_bezier(signal, variance=0.05):
    n = len(signal)
    t = np.linspace(0, 1, n)
    _, y_vals = rand_knots(k=4, variance=variance)
    bezier_map = bezier_curve(t, y_vals)
    bezier_map = np.clip(bezier_map, 0, 1)
    return np.interp(t, bezier_map, signal)

def twarp_pchip(signal, k=4, variance=0.05):
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, y_vals = rand_knots(k=k, variance=variance)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    pchip_map = np.clip(pchip_map, 0, 1)
    return np.interp(t, pchip_map, signal)

def awarp_pchip(signal, k=4, variance=0.05):
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, _ = rand_knots(k=k, variance=variance)
    y_vals = np.random.uniform(-variance, variance, size=k)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    return signal + pchip_map

def awarp_bezier(signal, variance=0.05):
    n = len(signal)
    t = np.linspace(0, 1, n)
    y_vals = np.random.uniform(-variance, variance, size=4)
    bezier_map = bezier_curve(t, y_vals)
    return signal + bezier_map

def amod_bezier(signal, variance=0.05):
    n = len(signal)
    t = np.linspace(0, 1, n)
    y_vals = np.random.uniform(1-variance, 1+variance, size=4)
    bezier_map = bezier_curve(t, y_vals)
    return signal * bezier_map


def amod_pchip(signal, k=4, variance=0.05):
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, _ = rand_knots(k=k, variance=variance)
    y_vals = np.random.uniform(1-variance, 1+variance, size=k)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    return signal * pchip_map



# X=np.load("../datasets/Stanford/X_test.npy")
# fs = 40

# x=X[1,:]
# t = np.arange(len(x)) / fs  # Make sure you set the correct fs
# x_pchip = awarp_pchip(x, k=4, variance=0.05)
# x_pchip = twarp_pchip(x_pchip, k=4, variance=0.01)

# x_bezier = awarp_bezier(x, variance=0.05)
# x_bezier = twarp_bezier(x_bezier, variance=0.01)

# # Plot both signals
# plt.figure(figsize=(10, 4))
# plt.plot(t, x, label='Original PPG')
# plt.plot(t, x_pchip, label='PCHIP Amp & Time Warped PPG')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Augmentation effect")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()




# # Plot both signals
# plt.figure(figsize=(10, 4))
# plt.plot(t, x, label='Original PPG')
# plt.plot(t, x_bezier, label='Bezier Amp & Time Warped PPG')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Augmentation effect")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

