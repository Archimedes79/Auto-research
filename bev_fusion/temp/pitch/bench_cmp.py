import sys, time, numpy as np
sys.path[:0] = [".", "../autoresearch"]
from evalsuite.comparison import maps as M
import cv2
rng = np.random.RandomState(0)
H, W, S = 120, 104, 12                      # grid, number of valid selections per layer (range bins x known/unknown ...)
g = rng.rand(H, W) < 0.08; p = np.roll(g, 1, axis=1) | (rng.rand(H, W) < 0.02)
valid = [rng.rand(H, W) < 0.5 for _ in range(S)]
def t(f, n=200):
    f(); t0 = time.perf_counter()
    for _ in range(n): r = f()
    return (time.perf_counter() - t0) / n * 1000, r
tol = 2.0                                      # 1 m = 2 cells
ms_d, near_g = t(lambda: M.dilate(g, tol))
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
disc = np.zeros((5, 5), np.uint8)
for dx in range(-2, 3):
    for dy in range(-2, 3):
        disc[dx + 2, dy + 2] = dx * dx + dy * dy <= 4
ms_c, near_c = t(lambda: cv2.dilate(g.astype(np.uint8), disc).astype(bool))
print(f"dilate by 2 cells: numpy loop {ms_d:.3f} ms, cv2.dilate {ms_c:.3f} ms, identical: {bool((near_g == near_c).all())}")
near_p = M.dilate(p, tol)
ms_a, ref = t(lambda: np.array([M.tolerant_counts(p, g, near_g, near_p, v) for v in valid]))
V = np.stack([v.ravel() for v in valid]).astype(np.float32)             # (S, N)
def fast():
    X = np.stack([(p & near_g).ravel(), (p & ~near_g).ravel(), (g & near_p).ravel(), (g & ~near_p).ravel()], 1).astype(np.float32)   # (N, 4)
    return np.rint(V @ X).astype(np.int64)
ms_b, new = t(fast)
print(f"counts for {S} selections: {ms_a:.3f} ms (loop of tolerant_counts)  vs {ms_b:.3f} ms (one matrix product), identical: {bool((ref == new).all())}")
