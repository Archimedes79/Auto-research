import sys

mode, path = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


if mode == "interface":
    rep("| `predict(network, frames, ...)` | `predict NETWORK [FRAMES]` | run the network on what is not stored yet and store it |",
        "| `predict(network, frames, taps, ...)` | `predict NETWORK [FRAMES] --taps a,b` | run the network on what is not stored yet and store it, with the intermediate results `taps` when asked (a frame stored without a tap is predicted again to get it); warnings of the memory's health (D70: empty at the scene start, filling after) |\n| `tap(results, frame, name)` | `tap RESULTS FRAME [--name N]` | which taps a stored frame has, or the file, shape and dtype of one |")
elif mode == "bev":
    rep("The val loss is computed by the network while it\npredicts,",
        "Intermediate results (`--taps lift,fused,motion_l2,...`: the taps of `bev_adapter.bev_hooks()`, `trunk`, ...) are stored next to the frames under `taps/` and read with `Evaluation.tap`;\nthe health of the camera memory (D70) is checked on every scene the network runs from the tap `memory_conf` (empty at the first frame, > 0.05 = not initialised; not filling = not used) and shows\nas a warning. What the interface does not do yet: a ground-truth-free `predict` (the Subject computes its val loss from the GT, so a frame without labels needs a Subject without the loss: a change\nof the frozen `bev_adapter.py` or a second Subject class). The val loss is computed by the network while it\npredicts,")
open(path, "w", encoding="utf-8").write(s)
print("ok")
