import numpy as np


def deep_eq(a, b):
    """exact equality of nested dicts / lists / tuples / arrays / numbers (nan == nan)"""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float))))) if a.dtype.kind in "fiub" and b.dtype.kind in "fiub" else a.tolist() == b.tolist()
    if isinstance(a, float) or isinstance(b, float):
        return (a == b) or (a != a and b != b)
    return a == b
