"""E387 launch gate check (card section 1, the critic's B1): the step-100 / 200 rows of the three gate runs against the stored E368 rows. (i) data order: gt_links / pos_pairs / gt_outside of every gate run equal the stored z / r rows exactly; (ii) total and nll_s of the z and r gate runs within 0.05 of the stored rows; (iii) f: s per step, peak GPU memory, u. Prints PASS / FAIL per check and a final line."""
import json
import os

R = os.environ.get("BFLC_ROOT", ".")


def rows(run):
    out = {}
    for ln in open(os.path.join(R, "runs", run, "log.jsonl"), encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r.get("step")] = r
    return out


stored = {"z": rows("e368_z_s0"), "r": rows("e368_r_s0")}
gate = {a: rows(f"e387_gate_{a}_s0") for a in ("z", "r", "f")}
ok_all = True
for a in ("z", "r", "f"):
    for st in (100, 200):
        assert st in gate[a], f"gate run {a}: no row of step {st} (the timeout was too short: {sorted(gate[a])[-3:]})"
for a in ("z", "r", "f"):
    for st in (100, 200):
        for k in ("gt_links", "pos_pairs", "gt_outside"):
            ref = stored["z"][st][k]
            good = abs(gate[a][st][k] - ref) < 1e-3
            ok_all &= good
            print(f"{'PASS' if good else 'FAIL'} (i) data order {a} step {st} {k}: {gate[a][st][k]} against the stored {ref}")
for a in ("z", "r"):
    for st in (100, 200):
        for k in ("total", "nll_s"):
            d = gate[a][st][k] - stored[a][st][k]
            good = abs(d) <= 0.05
            ok_all &= good
            print(f"{'PASS' if good else 'FAIL'} (ii) {a} step {st} {k}: gate {gate[a][st][k]:.4f} stored {stored[a][st][k]:.4f} (diff {d:+.4f}, limit 0.05)")
f = gate["f"]
last = f[max(f)]
print(f"(iii) f: steps logged up to {max(f)}, s/step {last.get('s_per_step')}, gpu_gb {last.get('gpu_gb')}, u {last.get('u')}, u_none {last.get('u_none')}, total step 100 {f[100]['total']:.4f} step 200 {f[200]['total']:.4f}")
print("GATE", "PASS" if ok_all else "FAIL: (i) stops the card, (ii) re-trains z_s0 on the landed code (card section 1)")
