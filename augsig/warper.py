import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import comb
import warnings


def rand_knots(k=4, variance=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    var_max = 1 / (2 * k - 4)

    if variance <= 0:
        variance = 0.01 * var_max
        warnings.warn(
            "The variance was not a positive value! "
            "It is automatically set to (0.01 * maximum value)!",
            UserWarning,
        )

    if variance >= var_max:
        variance = 0.95 * var_max
        warnings.warn(
            "The variance was too large! "
            "It is automatically set to (0.95 * maximum value)!",
            UserWarning,
        )

    x_vals = np.append(np.append(0, np.arange(0, 1, 2 * var_max) + var_max), 1)
    y_vals = x_vals.copy()

    x_vals[1:-1] += rng.uniform(-variance, variance, size=k - 2)
    y_vals[1:-1] += rng.uniform(-variance, variance, size=k - 2)
    return x_vals, y_vals


def bezier_cubic_curve(t, points):
    P0, P1, P2, P3 = points
    return (
        (1 - t)**3 * P0 +
        3 * (1 - t)**2 * t * P1 +
        3 * (1 - t) * t**2 * P2 +
        t**3 * P3
    )


def bezier_curve(t, control_points):
    """
    Generalized Bezier curve for any number of control points.

    Args:
        t (np.ndarray): Parameter values in [0, 1], shape (N,)
        control_points (np.ndarray or list): Control points, shape (K,)

    Returns:
        np.ndarray: Bezier curve values evaluated at each t, shape (N,)
    """
    control_points = np.asarray(control_points)
    n = len(control_points) - 1
    curve = np.zeros_like(t, dtype=float)

    for i in range(n + 1):
        binomial = comb(n, i)
        curve += binomial * (1 - t) ** (n - i) * t ** i * control_points[i]

    return curve


def twarp_bezier(signal, k=4, variance=0.05, rng=None):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    _, y_vals = rand_knots(k=k, variance=variance, rng=rng)
    bezier_map = bezier_curve(t, y_vals)
    bezier_map = np.clip(bezier_map, 0, 1)
    return np.interp(t, bezier_map, signal)


def twarp_pchip(signal, k=4, variance=0.05, rng=None):
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, y_vals = rand_knots(k=k, variance=variance, rng=rng)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    pchip_map = np.clip(pchip_map, 0, 1)
    return np.interp(t, pchip_map, signal)


def adrift_pchip(signal, k=4, variance=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, _ = rand_knots(k=k, variance=variance, rng=rng)
    y_vals = rng.uniform(-variance, variance, size=k)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    return signal + pchip_map


def adrift_bezier(signal, k=4, variance=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    y_vals = rng.uniform(-variance, variance, size=k)
    bezier_map = bezier_curve(t, y_vals)
    return signal + bezier_map


def amod_bezier(signal, k=4, variance=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    y_vals = rng.uniform(1 - variance, 1 + variance, size=k)
    bezier_map = bezier_curve(t, y_vals)
    return signal * bezier_map


def amod_pchip(signal, k=4, variance=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)
    x_vals, _ = rand_knots(k=k, variance=variance, rng=rng)
    y_vals = rng.uniform(1 - variance, 1 + variance, size=k)
    pchip_map = PchipInterpolator(x_vals, y_vals)(t)
    return signal * pchip_map


def drift_linear(signal, a, b, rng=None):
    """
    Add linear baseline drift to a signal: y = a*t + b

    Args:
        signal (np.ndarray): input 1D signal
        a (float or list/tuple of length 2): slope, or interval [a1, a2]
        b (float or list/tuple of length 2): intercept, or interval [b1, b2]
        rng (np.random.Generator, optional): random generator

    Returns:
        np.ndarray: signal with added linear drift
    """
    if rng is None:
        rng = np.random.default_rng()

    signal = signal.squeeze()
    n = len(signal)
    t = np.linspace(0, 1, n)

    if isinstance(a, (list, tuple)) and len(a) == 2:
        a_val = rng.uniform(a[0], a[1])
    elif isinstance(a, (int, float)):
        a_val = a
    else:
        raise ValueError("a must be a scalar or a length-2 interval")

    if isinstance(b, (list, tuple)) and len(b) == 2:
        b_val = rng.uniform(b[0], b[1])
    elif isinstance(b, (int, float)):
        b_val = b
    else:
        raise ValueError("b must be a scalar or a length-2 interval")

    drift = a_val * t + b_val
    return signal + drift
