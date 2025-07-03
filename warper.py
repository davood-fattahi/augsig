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

def twarp_bezier(signal, k=4, variance=0.05):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    _, y_vals = rand_knots(k=k, variance=variance)
    bezier_map = bezier_curve(t, y_vals)
    bezier_map = np.clip(bezier_map, 0, 1)
    return np.interp(t, bezier_map, signal)

def twarp_pchip(signal, k=4, variance=0.05):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, y_vals = rand_knots(k=k, variance=variance)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    pchip_map = np.clip(pchip_map, 0, 1)
    return np.interp(t, pchip_map, signal)

def adrift_pchip(signal, k=4, variance=0.05):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, _ = rand_knots(k=k, variance=variance)
    y_vals = np.random.uniform(-variance, variance, size=k)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    return signal + pchip_map

def adrift_bezier(signal, k=4, variance=0.05):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    y_vals = np.random.uniform(-variance, variance, size=k)
    bezier_map = bezier_curve(t, y_vals)
    return signal + bezier_map

def amod_bezier(signal, k=4, variance=0.05):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    y_vals = np.random.uniform(1-variance, 1+variance, size=k)
    bezier_map = bezier_curve(t, y_vals)
    return signal * bezier_map


def amod_pchip(signal, k=4, variance=0.05):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, _ = rand_knots(k=k, variance=variance)
    y_vals = np.random.uniform(1-variance, 1+variance, size=k)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    return signal * pchip_map


