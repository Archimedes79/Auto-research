"""One-off: error of the small-omega Taylor expansions of the ego-motion (constant turn rate and velocity) splat positions.

p' = R(-w dt) (p + v_obj dt - d(dt)),  d = (v/w sin(w dt), v/w (1 - cos(w dt)))   (exact)
first order in w:  x' = x + (vox - v) dt + w dt (y + voy dt)
                   y' = y + voy dt - w dt (x + vox dt) + 1/2 v w dt^2
second order adds: x' += -1/2 w^2 dt^2 (x + vox dt) + 1/6 v w^2 dt^3
                   y' += -1/2 w^2 dt^2 (y + voy dt)
Run: wsl python3 temp/setup/check_ctrv_taylor.py
"""
import math


def exact(x, y, dt, v, w, vox=0.0, voy=0.0):
    th = w * dt
    if abs(w) < 1e-12:
        dx, dy = v * dt, 0.0
    else:
        dx, dy = v / w * math.sin(th), v / w * (1 - math.cos(th))
    qx, qy = x + vox * dt - dx, y + voy * dt - dy
    c, s = math.cos(th), math.sin(th)
    return c * qx + s * qy, -s * qx + c * qy


def taylor(x, y, dt, v, w, vox=0.0, voy=0.0, order=2):
    xp = x + (vox - v) * dt + w * dt * (y + voy * dt)
    yp = y + voy * dt - w * dt * (x + vox * dt) + 0.5 * v * w * dt * dt
    if order >= 2:
        xp += -0.5 * w * w * dt * dt * (x + vox * dt) + v * w * w * dt ** 3 / 6
        yp += -0.5 * w * w * dt * dt * (y + voy * dt)
    return xp, yp


cases = [  # (name, v m/s, w rad/s, dt s)
    ("radar -> camera, highway", 30.0, 0.05, 0.08),
    ("camera history, highway", 30.0, 0.05, 0.5),
    ("camera history, city turn", 8.0, 0.5, 0.5),
    ("1 s history, city turn", 8.0, 0.5, 1.0),
    ("hard: fast and turning", 30.0, 0.2, 0.5),
]
points = [(60.0, 0.0), (30.0, 20.0), (5.0, -20.0)]
print("case | v | w | dt | max position error over 3 points (m): 1st order / 2nd order")
for name, v, w, dt in cases:
    e1 = e2 = 0.0
    for x, y in points:
        ex = exact(x, y, dt, v, w, 3.0, -2.0)
        for order in (1, 2):
            t = taylor(x, y, dt, v, w, 3.0, -2.0, order)
            e = math.hypot(t[0] - ex[0], t[1] - ex[1])
            if order == 1:
                e1 = max(e1, e)
            else:
                e2 = max(e2, e)
    print(f"{name:27s} | {v:4.0f} | {w:4.2f} | {dt:4.2f} | {e1:7.4f} / {e2:7.4f}")

# chained first-order steps of 1/12 s (one camera frame) against one exact jump over the same total time
print("\nchained 1st-order steps of 1/12 s vs exact, static points: case | total dt | max error (m)")
for name, v, w, total in [("highway", 30.0, 0.05, 0.5), ("fast and turning", 30.0, 0.2, 0.5),
                          ("city turn", 8.0, 0.5, 0.5), ("city turn", 8.0, 0.5, 1.0), ("tight turn", 5.0, 1.0, 0.5)]:
    n = int(round(total * 12))
    err = 0.0
    for x, y in points:
        px, py = x, y
        for _ in range(n):
            px, py = taylor(px, py, total / n, v, w, 0.0, 0.0, order=1)
        ex = exact(x, y, total, v, w)
        err = max(err, math.hypot(px - ex[0], py - ex[1]))
    print(f"{name:17s} v={v:4.0f} w={w:4.2f} | {total:4.2f} s in {n} steps | {err:7.4f}")
