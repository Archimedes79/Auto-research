"""The hashed part of the toy ground truth: the disc mask of an annotation. An edit of this file (or another option) is another signature and another store."""
import numpy as np


def rasterise(ann, size, supersample=4):
    """(N, size, size) bool: a pixel is inside when at least half of its supersample x supersample sub-pixel centres lie in the disc (cx, cy, r) of its frame"""
    n = int(supersample)
    centres = ((np.arange(size)[:, None] + (np.arange(n)[None, :] + 0.5) / n)).reshape(-1)         # size * n sub-pixel centres along one axis
    out = np.zeros((len(ann), size, size), bool)
    for k, (cx, cy, r) in enumerate(np.asarray(ann, np.float64)):
        inside = (centres[None, :] - cx) ** 2 + (centres[:, None] - cy) ** 2 <= r * r           # rows = y, columns = x
        out[k] = inside.reshape(size, n, size, n).mean(axis=(1, 3)) >= 0.5
    return out
