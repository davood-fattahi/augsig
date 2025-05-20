import numpy as np
def normalize_per_instance(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)
    return x_norm