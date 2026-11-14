"""one-off: coordinates for the 'targets by location' picture (pure python): 8 x 8 cells of 40 px (= 2 m), a stem that splits into two branches"""
import math
S = 40; X0, Y0 = 40, 40
pieces = {
    "stem": [(190, 360), (190, 200)],
    "left": [(190, 200), (110, 40)],
    "right": [(190, 200), (270, 40)],
}
def sample(a, b, n=400):
    return [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in range(n + 1)]
split_cell = None
items = {}
for name, (a, b) in pieces.items():
    pts = sample(a, b)
    best = {}
    for k, (x, y) in enumerate(pts):
        c = (int((x - X0) // S), int((y - Y0) // S))
        if not (0 <= c[0] < 8 and 0 <= c[1] < 8):
            continue
        cx, cy = X0 + c[0] * S + S / 2, Y0 + c[1] * S + S / 2
        score = -1 if k in (0, len(pts) - 1) else math.hypot(x - cx, y - cy)
        if c not in best or score < best[c][0]:
            best[c] = (score, k, x, y)
    items[name] = sorted(best.items(), key=lambda kv: kv[1][1])
split_cell = (int((190 - X0) // S), int((200 - Y0) // S))
print("split cell", split_cell)
for name, it in items.items():
    print(name, [(c, round(v[2]), round(v[3])) for c, v in it])
