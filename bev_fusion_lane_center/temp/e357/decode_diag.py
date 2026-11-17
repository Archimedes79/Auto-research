"""E357 T0, requested by the lane lead (03:35): WHY does the head deliver duplicate / false links? A CPU check on a stored dump, NOT a verdict and NOT a delivered rule (a mutual-best or top-1 decode would be a new output selection, D88; IN-SAMPLE on VAL).
Precision / recall / F1 of the pair probabilities against the GT links, separately for the SHORT links (chord <= 8 m) and the LONG links (chord > 8 m; every long GT link is a junction link at 2.5 m), under five decodes:
  thr      every candidate with p >= thr (the card's read-out)
  src1     per source node the highest-p candidate, kept when p >= thr
  mutual   a candidate that is the highest-p one of its source AND of its destination, kept when p >= thr
  src1gt   (oracle on the SOURCE: only the nodes that really have a link of that class) the highest-p candidate of those: P = R by construction, the ranking quality given the source
  thr_gt   (oracle on the source) every candidate with p >= thr from the nodes that really have a link of that class: what the over-production would be if the node-level question were answered
usage (from the problem folder): wsl bash tools/env/_py14.sh temp/e357/decode_diag.py <run>"""
import os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
import jtok_eval as JE  # noqa: E402

run = sys.argv[1]
with open(os.path.join(JE.ROOT, "runs", run, "tok_val.pkl"), "rb") as fh:
    preds = pickle.load(fh)
frames, table = JE.val_tables(2.5)
assert len(preds) == 873
THR = (0.5, 0.9)
DEC = ("thr", "src1", "mutual", "thr_gt", "src1gt")
acc = {(c, d, th): np.zeros(3) for c in ("short", "long") for d in DEC for th in THR}     # tp, predicted, gt
with JE.JT.lattice(2.5):
    for k, p in enumerate(preds):
        t = table(k)
        lk = p["links"]
        src, dst, pp = np.asarray(lk["src"], int), np.asarray(lk["dst"], int), np.asarray(lk["p"], float)
        pos = t["pos"].astype(float)
        ch = np.linalg.norm(pos[src] - pos[dst], axis=1)
        gtset = {"short": set(), "long": set()}
        for s, d, lg in zip(t["src"], t["dst"], t["long"]):
            gtset["long" if lg else "short"].add((int(s), int(d)))
        for cls, sel in (("short", ch <= JE.JT.SHORT), ("long", ch > JE.JT.SHORT)):
            s_, d_, p_ = src[sel], dst[sel], pp[sel]
            G = gtset[cls]
            gsrc = {s for s, _ in G}
            bs, bd = {}, {}
            for i, (s, d, q) in enumerate(zip(s_.tolist(), d_.tolist(), p_.tolist())):
                if s not in bs or q > p_[bs[s]]:
                    bs[s] = i
                if d not in bd or q > p_[bd[d]]:
                    bd[d] = i
            best_src = np.zeros(len(s_), bool)
            best_src[list(bs.values())] = True
            best_dst = np.zeros(len(s_), bool)
            best_dst[list(bd.values())] = True
            in_gt_src = np.array([s in gsrc for s in s_.tolist()], bool) if len(s_) else np.zeros(0, bool)
            for th in THR:
                for dec in DEC:
                    if dec == "thr":
                        m = p_ >= th
                    elif dec == "src1":
                        m = best_src & (p_ >= th)
                    elif dec == "mutual":
                        m = best_src & best_dst & (p_ >= th)
                    elif dec == "thr_gt":
                        m = in_gt_src & (p_ >= th)
                    else:
                        m = in_gt_src & best_src
                    tp = sum((int(s), int(d)) in G for s, d in zip(s_[m].tolist(), d_[m].tolist()))
                    acc[(cls, dec, th)] += (tp, int(m.sum()), len(G))
print(f"run {run}: decodes of the stored pair probabilities against the GT links (IN-SAMPLE on VAL; a diagnosis next to the verdict, never the verdict)")
print("class  decode   thr     predicted     P      R      F1")
for cls in ("short", "long"):
    for dec in DEC:
        for th in THR:
            if dec == "src1gt" and th != THR[0]:
                continue
            tp, npred, ngt = acc[(cls, dec, th)]
            P, R = tp / max(npred, 1), tp / max(ngt, 1)
            F = 2 * P * R / max(P + R, 1e-12)
            print(f"{cls:6s} {dec:8s} {th:<5g} {npred:11.0f}   {P:.3f}  {R:.3f}  {F:.3f}   (GT {ngt:.0f})")
