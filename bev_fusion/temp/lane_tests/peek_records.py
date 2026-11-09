"""scratch: what a stored scene_records pickle holds (are there probability maps?)."""
import pickle
import sys

p = sys.argv[1]
d = pickle.load(open(p, "rb"))
print(type(d), len(d) if hasattr(d, "__len__") else "")


def show(x, depth=0, key=""):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{key} dict({len(x)}): {list(x)[:8]}")
        if depth < 3:
            for k in list(x)[:3]:
                show(x[k], depth + 1, str(k))
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{key} {type(x).__name__}({len(x)})")
        if depth < 3 and x:
            show(x[0], depth + 1, "[0]")
    else:
        sh = getattr(x, "shape", None)
        print(f"{pad}{key} {type(x).__name__} {sh if sh is not None else str(x)[:60]}")


show(d)
