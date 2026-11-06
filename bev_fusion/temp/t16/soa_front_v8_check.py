"""gate (c) on a DualBEV record (full or a test subset): its GT boxes per frame against the GT boxes inside the stored records of t16 / t17 (same token);
--fake: also write a stand-in LABEL pickle (the t16 detections, 1000 frames) to dry-run the `bev_suite.py front` command line before the real record exists.
usage: soa_front_v8_check.py RECORD.pkl [--fake temp/t16/soa/soa_front_v8_fake.pkl]"""
import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soa_front_v8 as S  # noqa: E402  (chdir to the bev_fusion folder)

rec = pickle.load(open(sys.argv[1], "rb"))
meta = rec["meta"]
rows = S.ref_rows()
pos, where = {}, {}
for r in rows:
    where[r["key"]] = (r["scene"], pos.get(r["scene"], 0))
    pos[r["scene"]] = pos.get(r["scene"], 0) + 1
refs = {run: pickle.load(open(os.path.join(run, "eval", "scene_records_n1000.pkl"), "rb")) for run in (S.REF, S.OTHER)}
mine, p2 = {}, {}
for tok in meta["frames"]:                         # the record's own frames in its own order, per scene
    sc, _ = where[tok]
    mine[tok] = rec["det"][sc][0.0][p2.get(sc, 0)][1]
    p2[sc] = p2.get(sc, 0) + 1
bad = {run: sum(int(mine[t] != refs[run]["det"][where[t][0]][0.0][where[t][1]][1]) for t in mine) for run in refs}
print(f"gate (c) on {sys.argv[1]}: {len(mine)} frames, GT signature {meta['gt_signature']} (ref {S.gt_signature()}), "
      f"frames whose GT boxes differ from the stored records of {list(refs)}: {bad}, "
      f"boxes {sum(len(v) for v in mine.values())} -> {'PASS' if not any(bad.values()) and meta['gt_signature'] == S.gt_signature() else 'FAIL'}")
if "--fake" in sys.argv:
    out = sys.argv[sys.argv.index("--fake") + 1]
    r16 = refs[S.REF]
    with open(out, "wb") as f:
        pickle.dump({"seg": {}, "det": {s: {0.0: [(d, [dict(g) for g in gs]) for d, gs in v[0.0]]} for s, v in r16["det"].items()},
                     "detr": {s: {} for s in r16["det"]}, "meta": dict(r16["meta"])}, f)
    print("stand-in written:", out)
