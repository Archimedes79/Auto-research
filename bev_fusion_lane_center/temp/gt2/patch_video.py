p = "tools/viz/junction_view.py"
s = open(p, encoding="utf-8").read()
old = "def _selftest():"
new = '''def render_scene(scene, out_dir, pred=None, every=1, frames=0, fps=2):
    """one mp4 (`<scene name>_junction.mp4`, mp4v, `fps` frames per second) of the keyframes of a VAL scene (id or name), every `every`-th, at most `frames` (0 = all); the same figure as `render`"""
    import cv2
    import qgraph_view as QV
    sid, name, toks = QV.resolve_scene(scene)
    toks = toks[::every]
    if frames:
        toks = toks[:frames]
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_frames")
    writer = None
    path = os.path.join(out_dir, f"{name}_junction.mp4")
    for t in toks:
        img = cv2.imread(render(t, tmp, pred))
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    writer.release()
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return path


def _selftest():'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''    p.add_argument("--pred", default=None)
    a = ap.parse_args()''', '''    p.add_argument("--pred", default=None)
    v = sub.add_parser("scene")
    v.add_argument("what", help="VAL scene id or name")
    v.add_argument("--out", required=True)
    v.add_argument("--pred", default=None)
    v.add_argument("--every", type=int, default=1)
    v.add_argument("--frames", type=int, default=0)
    a = ap.parse_args()''', 1)
s = s.replace('''    if a.selftest:
        _selftest()
    else:''', '''    if a.selftest:
        _selftest()
    elif a.cmd == "scene":
        print("written", render_scene(a.what, a.out, a.pred, a.every, a.frames))
    else:''', 1)
s = s.replace("    python junction_view.py --selftest\n", "    python junction_view.py scene <VAL scene id | name> --out DIR [--pred RUN] [--every K] [--frames N]       an mp4 of the keyframes (2 per second)\n    python junction_view.py --selftest\n", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
