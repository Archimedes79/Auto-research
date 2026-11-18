"""Mutation check of the comparison core: break ONE thing at a time in a COPY of the suite and see that `selftest comparison` fails.
A mutant that survives is a missing check. Slow (about 8 s per mutant); run it when the core changes:

    python Evaluation/comparison/mutation_check.py          (from autoresearch/, any python with numpy, matplotlib, scipy, shapely)

Each mutant is (file, exact text, replacement, what is broken); the text must occur exactly once. Add a mutant when you add a rule."""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))        # the folder that contains Evaluation/
CORE = os.path.join("Evaluation", "comparison")

MUTANTS = [
    ("maps.py", "(p & ~g & v).sum(), (~p & g & v).sum(), (~p & ~g & v).sum()], dtype=np.int64)\n\n\ndef scores",
     "(~p & g & v).sum(), (p & ~g & v).sum(), (~p & ~g & v).sum()], dtype=np.int64)\n\n\ndef scores", "confusion: fp and fn swapped"),
    ("maps.py", '"iou": ratio(tp, tp + fp + fn), "precision": p', '"iou": ratio(tp, tp + fp), "precision": p', "scores: iou without fn"),
    ("maps.py", "return 2.0 * precision * recall / (precision + recall) if", "return precision * recall / (precision + recall) if", "f1: not doubled"),
    ("maps.py", "<= radius_cells * radius_cells + 1e-9", "< radius_cells * radius_cells - 1e-9", "dilate: disc open"),
    ("maps.py", "cv2.dilate(np.ascontiguousarray(m, np.uint8), _disc(radius_cells).astype(np.uint8))", "cv2.dilate(np.ascontiguousarray(m, np.uint8), np.ones((3, 3), np.uint8))",
     "dilate: OpenCV with a square instead of the disc"),
    ("maps.py", "cols += [(p & near_g).ravel(), (p & ~near_g).ravel(), (g & near_p).ravel(), (g & ~near_p).ravel()]",
     "cols += [(p & near_p).ravel(), (p & ~near_p).ravel(), (g & near_g).ravel(), (g & ~near_g).ravel()]", "tolerant_counts_many: dilations swapped"),
    ("maps.py", "return out.reshape(len(valids), len(nears), 4).transpose(1, 0, 2)", "return out.reshape(len(nears), len(valids), 4)", "tolerant_counts_many: tolerance and selection axes mixed"),
    ("maps.py", "np.stack(cols, 1).astype(np.float32)", "np.stack(cols, 1).astype(np.float16)", "tolerant_counts_many: counts in half precision"),
    ("maps.py", "[(p & near_g & v).sum(), (p & ~near_g & v).sum(), (g & near_p & v).sum(), (g & ~near_p & v).sum()]",
     "[(p & near_p & v).sum(), (p & ~near_p & v).sum(), (g & near_g & v).sum(), (g & ~near_g & v).sum()]", "tolerant: dilations swapped"),
    ("maps.py", "tp = (p_hit + g_hit) / 2.0", "tp = p_hit", "tolerant iou: tp is not the mean"),
    ("maps.py", 'if a.dtype.kind == "f":', "if False:", "float maps accepted"),
    ("maps.py", "return float(num) / float(den) if den else NAN", "return float(num) / float(den) if den else 0.0", "ratio: 0 instead of nan"),
    ("maps.py", "    _same_shape(p, g, v)\n    return np.array([(p & g & v)", "    return np.array([(p & g & v)", "confusion: shape check removed"),
    ("maps.py", "(g, p) if tol_cells == 0 else", "(p, g) if tol_cells == 0 else", "tolerant_confusion: tolerance 0 uses the maps swapped"),
    ("boxes.py", "            if dist <= best_dist:", "            if dist < best_dist:", "match: threshold / tie rule strict"),
    ("boxes.py", "key=lambda i: -scores[i]", "key=lambda i: scores[i]", "match: worst score first"),
    ("boxes.py", "            taken.add(best)\n", "            pass\n", "match: boxes can be taken twice"),
    ("boxes.py", 'aoe = abs(math.remainder(d["yaw"] - g["yaw"], 2 * math.pi))', 'aoe = abs(d["yaw"] - g["yaw"])', "aoe: no wrap"),
    ("boxes.py", "p = p[int(round(100 * min_recall)) + 1:] - min_precision", "p = p[int(round(100 * min_recall)):] - min_precision", "ap_nuscenes: recall bin kept"),
    ("boxes.py", "np.interp(np.linspace(0, 1, 101), rec, prec, right=0)", "np.interp(np.linspace(0, 1, 101), rec, prec)", "ap_nuscenes: no zero beyond last recall"),
    ("boxes.py", "    prec = np.maximum.accumulate(prec[::-1])[::-1]\n    return float(np.sum", "    return float(np.sum", "ap_all_points: no envelope"),
    ("boxes.py", "np.lexsort((flags[:, k], -scores))", "np.lexsort((-flags[:, k], -scores))", "ties: optimistic"),
    ("boxes.py", "for i, j in enumerate(matched[tp_threshold]):", "for i, j in enumerate(matched[thresholds[0]]):", "match_flags: errors at the wrong threshold"),
    ("boxes.py", "        flags[:, k] = [float(j >= 0) for j in matched[t]]", "        flags[:, k] = [float(j >= 0) for j in matched[thresholds[0]]]", "match_flags: flags of the first threshold for all"),
    ("boxes.py", "ase = 1.0 - inter / union if union > 0 else NAN", "ase = 1.0 - inter / (union + 1.0) if union > 0 else NAN", "ase: wrong union"),
    ("boxes.py", "ase, aoe = (1.0 - (r_min / r_max) ** 2 if r_max > 0 else NAN), NAN", "ase, aoe = (1.0 - (r_min / r_max) if r_max > 0 else NAN), NAN", "circle ase: not squared"),
    ("boxes.py", 'return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])', 'return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cx"])', "center_distance: wrong axis"),
    ("boxes.py", "row[name] = float(v.mean()) if len(v) else NAN", "row[name] = float(v.sum()) if len(v) else NAN", "mean error: sum"),
    ("boxes.py", 'if n_gt == 0:\n        return NAN\n    if len(tp_flags) == 0:\n        return 0.0\n    prec, rec = _precision_recall(tp_flags, n_gt)\n    p = np.interp',
     'if len(tp_flags) == 0:\n        return 0.0\n    prec, rec = _precision_recall(tp_flags, max(n_gt, 1))\n    p = np.interp', "ap_nuscenes: no ground truth is not nan"),
    ("metrics.py", "                    if row[k] == row[k]:\n", "                    if True:\n", "BoxMetric: nan of a class without ground truth enters the mean"),
    ("metrics.py", "return {k: float(c[1] + c[2] + c[3]) for k in self.headline}", "return {k: 1.0 for k in self.headline}", "MapMetric: support constant"),
    ("metrics.py", "if tp_threshold not in thresholds:", "if False:", "BoxMetric: a tp_threshold that is no threshold accepted"),
    # found by the independent review (2026-10-31) and closed in the tests
    ("metrics.py", '("mATE", ["ate"])', '("mATE", ["ase"])', "BoxMetric: mATE made from ase"),
    ("metrics.py", '("mAP_all", ["ap_all@%g" % t for t in self.thr])', '("mAP_all", ["ap@%g" % t for t in self.thr])', "BoxMetric: mAP_all made from ap"),
    ("metrics.py", '"mATE": "low"', '"mATE": "high"', "BoxMetric: mATE is better when high"),
    ("metrics.py", 'self.headline = ("mAP", "mATE")', 'self.headline = ("mAP",)', "BoxMetric: mATE is no headline number"),
    ("metrics.py", 'elif k in B.TP_ERROR_KEYS and row["n_gt"] > 0 and row["n_tp"] == 0:', "elif False:", "BoxMetric: a missed class drops out of mATE"),
    ("metrics.py", "return {k: float(c[1] + c[2] + c[3]) for k in self.headline}", "return {k: float(c[2] + c[3]) for k in self.headline}", "MapMetric: support without false positives"),
    ("metrics.py", "if len(maps) not in (2, 3):", "if False:", "MapMetric: extract of four values accepted"),
    ("metrics.py", 'return "" if tol == 0 else "@%g" % tol', 'return "" if tol == 0 else "@%d" % tol', "MapMetric: tolerance 1.5 gets the key @1"),
    ("boxes.py", "errors[i] = [e[k] for k in TP_ERROR_KEYS]", 'errors[i] = [e["ate"], e["aoe"], e["ase"], e["ave"], e["biou"]]', "match_flags: ase and aoe columns swapped"),
    ("boxes.py", "                v = v[v == v]\n", "                v = v\n", "detection_numbers: nan errors not skipped"),
    ("boxes.py", 'max(b["length"], b["width"]) / 2', 'b["length"] / 2', "_as_circle: radius from the length only"),
    ("boxes.py", '"ave": math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", NAN) - g.get("vy", NAN))',
     '"ave": math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", 0.0) - g.get("vy", NAN))', "tp_errors: a missing vy counts as 0"),
    ("boxes.py", "best, best_dist = -1, thr", "best, best_dist = -1, thr + 1e-9", "match: the threshold has a tolerance"),
    ("boxes.py", "if tp[-1] > n_gt:", "if False:", "ap: more true positives than ground truth accepted"),
    ("boxes.py", "if flags.shape != (len(scores), len(thresholds)) or errors.shape != (len(scores), len(TP_ERROR_KEYS)):", "if False:", "detection_numbers: mis-shaped flags accepted"),
    ("boxes.py", "if _finite_geometry(d) and _finite_geometry(g):", "if True:", "tp_errors: nan geometry reaches shapely"),
    ("boxes.py", "    frames = list(frames) ", "    frames = frames ", "match_frames: a one-shot iterator is read once"),
    ("maps.py", "return int(math.floor(radius_cells + 1e-9))", "return int(math.floor(radius_cells))", "dilate: no tolerance for float noise in the radius"),
    ("maps.py", "if tol_cells < 0:", "if False:", "tolerant_confusion: negative tolerance accepted"),
    ("maps.py", "_same_shape(p, g, near_g, near_p, v)", "_same_shape(p, g, v)", "tolerant_counts: dilations of another shape accepted"),
    ("maps.py", "(g & near_p & v).sum(), (g & ~near_p & v).sum()]", "(g & near_g & v).sum(), (g & ~near_g & v).sum()]", "tolerant_counts: g_hit from the wrong dilation"),
    ("maps.py", 'if a.dtype.kind in "iu" and a.size and (a.min() < 0 or a.max() > 1):', "if False:", "_as_mask: integer map with 255 accepted"),
    ("boxes.py", "v = errors[order, j][tp] ", "v = errors[:, j][flags[:, k] > 0] ", "detection_numbers: error means summed in row order, not rank order"),
    # velocity, moving against static (2026-10-31)
    ("boxes.py", "out[0 if s > moving else 1] += 1", "out[0 if s >= moving else 1] += 1", "velocity_counts: a ground-truth speed of exactly `moving` is moving"),
    ("boxes.py", "        k = 0 if s > moving else 1\n", "        k = 0 if s >= moving else 1\n", "velocity_counts: a pair at exactly `moving` is moving"),
    ("boxes.py", 'math.hypot(d["vx"], d["vy"]) < slow', 'math.hypot(d["vx"], d["vy"]) <= slow', "velocity_counts: a detection of exactly `slow` is slow"),
    ("boxes.py", "out[6 + k] += s\n", "out[6 + k] += e\n", "velocity_counts: the zero predictor's error is the model's error"),
    ("boxes.py", "out[4 + k] += e\n", "out[4 + k] += s\n", "velocity_counts: the model's error is the ground-truth speed"),
    ("boxes.py", "zip(dets, match(dets, gts, thr)):\n        if j < 0:", "zip(dets, match(dets, gts, thr * 10)):\n        if j < 0:", "velocity_counts: pairs matched at another threshold than the true positives"),
    ("boxes.py", "        if not (math.isfinite(e) and math.isfinite(s)):", "        if False:", "velocity_counts: a pair without a velocity is counted"),
    ("boxes.py", "        if math.isfinite(s):\n            out[0 if", "        if True:\n            out[0 if", "velocity_counts: a ground-truth box without a velocity is static"),
    ("boxes.py", '        s = math.hypot(g.get("vx", NAN), g.get("vy", NAN))\n        if math.isfinite(s):',
     '        s = math.hypot(g.get("vx", NAN), g.get("vy", 0.0))\n        if math.isfinite(s):', "velocity_counts: half a ground-truth velocity (vy missing) counts as vy 0"),
    ("boxes.py", '        e = math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", NAN) - g.get("vy", NAN))',
     '        e = math.hypot(d.get("vx", 0.0) - g.get("vx", NAN), d.get("vy", NAN) - g.get("vy", NAN))', "velocity_counts: half a detection velocity (vx missing) counts as vx 0"),
    ("boxes.py", '        e = math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", NAN) - g.get("vy", NAN))',
     '        e = math.hypot(d.get("vx", NAN) - g.get("vx", NAN), d.get("vy", 0.0) - g.get("vy", NAN))', "velocity_counts: half a detection velocity (vy missing) counts as vy 0"),
    ("boxes.py", 'row[f"ave_skill_{g}"] = float(1.0 - e / z) if z > 0 else NAN', 'row[f"ave_skill_{g}"] = float(e / z) if z > 0 else NAN', "velocity_numbers: skill is the error ratio, not 1 - ratio"),
    ("boxes.py", 'row["ave_slow_moving"] = float(c[8] / c[2]) if c[2] else NAN', 'row["ave_slow_moving"] = float(c[8] / c[3]) if c[3] else NAN', "velocity_numbers: slow share of the static pairs"),
    ("boxes.py", "if c.shape[-1:] != (len(VELOCITY_COUNT_KEYS),) or c.ndim > 2:", "if False:", "velocity_numbers: mis-shaped counts accepted"),
]


def main() -> int:
    stale = []                                                           # every mutant text must still occur exactly once: check all before the slow part
    for fn, old, _new, what in MUTANTS:
        n = open(os.path.join(ROOT, CORE, fn), encoding="utf-8").read().count(old)
        if n != 1:
            stale.append(f"{what!r}: the text occurs {n} times in {fn}")
    if stale:
        raise SystemExit("the core changed, update these mutants (each text must occur exactly once):\n  " + "\n  ".join(stale))
    work = tempfile.mkdtemp(prefix="mut_")
    survived = []
    for i, (fn, old, new, what) in enumerate(MUTANTS):
        root = os.path.join(work, f"m{i}")
        os.makedirs(root)
        shutil.copytree(os.path.join(ROOT, "Evaluation"), os.path.join(root, "Evaluation"), ignore=shutil.ignore_patterns("__pycache__", "eval_out"))
        path = os.path.join(root, CORE, fn)
        s = open(path, encoding="utf-8").read()
        if s.count(old) != 1:
            raise SystemExit(f"mutant {what!r}: the text occurs {s.count(old)} times in {fn}, it must occur once (the core changed: update the mutant)")
        open(path, "w", encoding="utf-8").write(s.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "Evaluation", "selftest", "comparison"], cwd=root, capture_output=True, text=True, timeout=300)
        caught = r.returncode != 0 and "FAIL " in r.stdout                  # a check failed; a crash of the mutant itself (a typo) does not count
        print(("caught   " if caught else "SURVIVED ") + what + ("" if caught or r.returncode == 0 else "  (it crashed instead of failing a check)"), flush=True)
        if not caught:
            survived.append(what)
        shutil.rmtree(root)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{len(MUTANTS) - len(survived)} of {len(MUTANTS)} mutants caught; survivors: {survived}")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
