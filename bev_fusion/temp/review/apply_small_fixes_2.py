"""Small verified fixes of the structure review 2026-10-30 (batch 2). Each replacement asserts that the old text is present once."""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))      # C:/Auto-research
BF = os.path.join(ROOT, "bev_fusion")


def edit(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    print("edited", os.path.relpath(path, ROOT))


# 1 evaluation/bev_suite.py: the legacy compare fallback ignored --tag (compared the cold pickles for --tag _warm); --state warmup exists
edit(f"{BF}/evaluation/bev_suite.py",
     'recs = [os.path.join(r, "eval", "scene_records.pkl") for r in (c.run_a, c.run_b)]',
     'recs = [os.path.join(r, "eval", f"scene_records{c.tag}.pkl") for r in (c.run_a, c.run_b)]')
edit(f"{BF}/evaluation/bev_suite.py",
     'bev_legacy.compare(c.run_a, c.run_b, c.boot, c.seed, c.det_a, c.det_b, c.dt)',
     'bev_legacy.compare(c.run_a, c.run_b, c.boot, c.seed, c.det_a, c.det_b, c.dt, c.tag)')
edit(f"{BF}/evaluation/bev_suite.py",
     'p.add_argument("--state", default="carry", choices=["carry", "reset"])',
     'p.add_argument("--state", default="carry", choices=["carry", "reset", "warmup"])')
edit(f"{BF}/evaluation/bev_legacy.py",
     'def compare(run_a, run_b, n_boot=1000, seed=0, det_a="det", det_b="det", dt=0.0):',
     'def compare(run_a, run_b, n_boot=1000, seed=0, det_a="det", det_b="det", dt=0.0, tag=""):')
edit(f"{BF}/evaluation/bev_legacy.py",
     '''    ra = pickle.load(open(os.path.join(run_a, "eval", "scene_records.pkl"), "rb"))
    rb = pickle.load(open(os.path.join(run_b, "eval", "scene_records.pkl"), "rb"))''',
     '''    ra = pickle.load(open(os.path.join(run_a, "eval", f"scene_records{tag}.pkl"), "rb"))    # tag: "" cold, "_warm" ...
    rb = pickle.load(open(os.path.join(run_b, "eval", f"scene_records{tag}.pkl"), "rb"))''')

# 2 evalsuite/render/video.py: OpenCV decodes MPEG-4 part 2 ('FMP4') too, so frames alone prove nothing about playability
edit(f"{ROOT}/autoresearch/evalsuite/render/video.py",
     '''    """decode the file again: all frames present, the first frame not black"""
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.max() == 0 or n != n_frames:
        raise RuntimeError(f"{path}: read back {n} frames (expected {n_frames}), first frame {'ok' if ok else 'unreadable'}")
''',
     '''    """decode the file again: all frames present, the first frame not black, the codec H.264"""
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    ok, frame = cap.read()
    cap.release()
    codec = "".join(chr((fcc >> 8 * i) & 255) for i in range(4)).lower()
    if not ok or frame is None or frame.max() == 0 or n != n_frames:
        raise RuntimeError(f"{path}: read back {n} frames (expected {n_frames}), first frame {'ok' if ok else 'unreadable'}")
    if codec not in ("h264", "avc1"):         # OpenCV decodes MPEG-4 part 2 ('FMP4') as well, which browsers refuse
        raise RuntimeError(f"{path}: codec {codec!r}, not H.264 - it would not play in a browser")
''')

# 3 training/losses.py: _wsum / _wcount are defined twice with identical bodies (loss_extras.py is the home)
src = open(f"{BF}/training/losses.py", encoding="utf-8").read()
assert src.count("from .loss_extras import boundary_fp_loss, dice_semimetric_loss, ramp") == 1
edit(f"{BF}/training/losses.py",
     '''def _wsum(x, w):
    """sum over batch and cells per channel: x (B, K, X, Y), w broadcastable -> (K,)."""
    return (x * w).sum((0, 2, 3))


def _wcount(w, like):
    return w.expand_as(like).sum((0, 2, 3)).clamp(min=EPS)


''', '')
edit(f"{BF}/training/losses.py",
     "from .loss_extras import boundary_fp_loss, dice_semimetric_loss, ramp",
     "from .loss_extras import _wcount, _wsum, boundary_fp_loss, dice_semimetric_loss, ramp")
edit(f"{BF}/training/losses.py",
     "from loss_extras import boundary_fp_loss, dice_semimetric_loss, ramp",
     "from loss_extras import _wcount, _wsum, boundary_fp_loss, dice_semimetric_loss, ramp")

# 4 plan.md: a patch script corrupted one line (a newline inside the file name)
edit(f"{BF}/plan.md",
     "information-preserving (Mirko, 2026-10-25): \nadar_encoding.md section A records",
     "information-preserving (Mirko, 2026-10-25): `radar_encoding.md` section A records")
