def normalize_per_instance(x):
    # x: shape (B, L, P)
    x_min = x.view(x.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    x_max = x.view(x.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)
    return x_norm