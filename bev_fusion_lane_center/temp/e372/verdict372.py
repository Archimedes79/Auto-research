"""E372 scratch: the numbers of the card's S4-HOLDS rule for one seed (section 0, v3.2), read from the stored files only.
usage (from the problem folder): python3 temp/e372/verdict372.py <seed>
Reads runs/e372_{z,s4,b}_s<S>/ (log.jsonl rows = means over 100 steps, status.json, the live score cards) and prints:
the paired differences s4 - z, s4 - ref, s4 - b, b - z with their scene-bootstrap intervals, the D113 floors, the kill checks
(head destroyed, B destroyed, overflow share, skipped / roll-backs), the update share, the moving-label flags (gt_cov, n_valid,
pair_cov of s4 against z on the same logged steps) and the B' proof fields of the cards."""
import json, sys, os

S = int(sys.argv[1])
CELL = "rec_lt0.3_ltl0.2_kt0.6_nng_live"
FLOORS = {0: dict(topo_F1=0.6359, **{"F1@1.0": 0.8182}, lanes_found=0.6066), 1: dict(topo_F1=0.6349, **{"F1@1.0": 0.8060}, lanes_found=0.5937)}[S]
KEYS = ("topo_F1", "junction_F1", "F1@1.0", "lanes_found")


def card(arm, ref):
    p = f"runs/e372_{arm}_s{S}/token_card_{CELL}_vs_{ref}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def rows(arm):
    return [json.loads(l) for l in open(f"runs/e372_{arm}_s{S}/log.jsonl")]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def show_pair(label, c):
    if c is None:
        print(f"  {label}: card missing"); return {}
    p = c["info"]["paired_vs_ref_run"]; out = {}
    for k in KEYS:
        d = p[k]; out[k] = d
        print(f"  {label:10s} {k:12s} {d['a']:.4f} - {d['b']:.4f} = {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    return out


print(f"=== E372 seed {S}")
ref = f"e373_cb_s{S}"
s4_ref = show_pair("s4 - ref", card("s4", ref))
s4_z = show_pair("s4 - z", card("s4", f"e372_z_s{S}"))
b_z = show_pair("b - z", card("b", f"e372_z_s{S}"))
z_ref = show_pair("z - ref", card("z", ref))
b_ref = show_pair("b - ref", card("b", ref))
# s4 - b from the two cards against z (same frames, same z): the difference of the a values; no interval (no card pairs s4 with b)
if s4_z and b_z:
    for k in KEYS:
        print(f"  s4 - b     {k:12s} {s4_z[k]['a'] - b_z[k]['a']:+.4f} (no paired interval: difference of two cards)")

c = card("s4", ref)
if c:
    m = c["metrics"]
    print("  floors (D113):", {k: (round(m[k], 4), FLOORS[k], m[k] >= FLOORS[k]) for k in FLOORS})
    i = c["info"]
    print("  b_source:", json.dumps(i.get("b_source"))[:400])
    print("  live:", json.dumps(i.get("live"))[:300], "| tokens_exact:", i.get("tokens_exact"))
for arm in ("z", "b"):
    cc = card(arm, ref)
    if cc:
        print(f"  {arm} b_source:", json.dumps(cc["info"].get("b_source"))[:300])

print("  kill checks (the trainer checks them on the per-step series; the rows' head / b_loss are single steps, b_start / b_mean / h_start are its 200-step means):")
for arm in ("z", "s4", "b"):
    if not os.path.exists(f"runs/e372_{arm}_s{S}/log.jsonl"):
        print(f"    {arm}: no log"); continue
    R = rows(arm)
    st = json.load(open(f"runs/e372_{arm}_s{S}/status.json"))
    bs = [r["b_start"] for r in R if r.get("b_start") == r.get("b_start")]
    bm = max(r["b_mean"] for r in R if r["step"] > 200)
    hs = [r["h_start"] for r in R if r.get("h_start") == r.get("h_start")]
    print(f"    {arm:3s} state {st['state']} steps {st['step']} skipped {st['skipped']} rollbacks {st['rollbacks']}; h_start {hs[-1]:.4f}; "
          f"b_start {bs[-1]:.4f}, largest logged b_mean after 200 {bm:.4f} (x{bm / bs[-1]:.3f}, kill > 1.20 in b / s4); overflow count {R[-1].get('overflow', 0.0):.0f}")

print("  update share (s4, every 200 steps):", [round(r["ushare"], 3) for r in rows("s4") if "ushare" in r])
Z = {r["step"]: r for r in rows("z")}; S4 = {r["step"]: r for r in rows("s4")}
common = sorted(set(Z) & set(S4))
for k, lim in (("gt_cov", "lower by > 0.02 abs"), ("n_valid", "differs > 10 %"), ("pair_cov", "differs > 10 %")):
    zs = mean([Z[s][k] for s in common]); ss = mean([S4[s][k] for s in common])
    print(f"  moving labels {k}: s4 {ss:.4f} vs z {zs:.4f} (diff {ss - zs:+.4f}, rel {(ss - zs) / zs:+.3%}; flag when {lim})")
