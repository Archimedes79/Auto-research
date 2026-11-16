"""scratch: patch the dev copy of jmaps_train.py: --no-paint / --no-zone (E332)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = "    wsl bash tools/env/_py14.sh jmaps_train.py --name e286_j0_s0 --seed 0 --base runs/e273_p_s0/model.pt [--ctx none|dil|attn] [--dirx] [--joint] [--steps 3000] [--smoke]\n"
b = "    wsl bash tools/env/_py14.sh jmaps_train.py --name e286_j0_s0 --seed 0 --base runs/e273_p_s0/model.pt [--ctx none|dil|attn] [--d 64] [--dirx] [--no-paint] [--no-zone] [--joint] [--steps 3000] [--smoke]\n"
assert a in s
s = s.replace(a, b)
a = '''    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.dirx:                                                                                           # the flag lives in the ctx string: jends_train.py, jmaps_eval.py and junction_view.py build JModel(ctx=args["ctx"])
        a.ctx = a.ctx + "+dirx"
'''
b = '''    ap.add_argument("--no-paint", action="store_true", help="E332: no 0.5 m paint branch (stop lines, crossings: -1.48 of 3.26 GMACs of the head), no paint loss; stored as ctx '<ctx>+nopaint'. The pair head must then run --no-evidence")
    ap.add_argument("--no-zone", action="store_true", help="E332: the zone / core maps are not supervised (nothing reads them without the evidence); stored as ctx '<ctx>+nozone'")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.dirx:                                                                                           # the flag lives in the ctx string: jends_train.py, jmaps_eval.py and junction_view.py build JModel(ctx=args["ctx"])
        a.ctx = a.ctx + "+dirx"
    if a.no_paint:
        a.ctx = a.ctx + "+nopaint"
    if a.no_zone:
        a.ctx = a.ctx + "+nozone"
'''
assert a in s
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("patched", p)
