import numpy as np
import qgraph_view as QV

orig_show = QV._show
def run(name):
    try:
        QV._selftest(); print("MUTANT SURVIVED:", name)
    except AssertionError as e:
        print("killed:", name, "->", str(e)[:90])
    finally:
        QV._show = orig_show

def show_notranspose(ax, img):
    ax.imshow(img, origin="lower", extent=(-50, 50, -50, 50), interpolation="nearest", zorder=0); return QV._bev(ax)
QV._show = show_notranspose
run("image not transposed (x and y swapped)")
def show_flip(ax, img):
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="upper", extent=(-50, 50, -50, 50), interpolation="nearest", zorder=0); return QV._bev(ax)
QV._show = show_flip
run("image flipped in y")
