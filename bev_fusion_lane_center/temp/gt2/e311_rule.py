"""the rule of the E311 card applied to temp/gt2/snap/*.json: per network and threshold, both seeds; KEEP when every seed of every confirmation network (e287_p2, e305_e) holds"""
import glob, json, collections
R = collections.defaultdict(dict)
for f in glob.glob("../../temp/gt2/snap/*.json"):
    d = json.load(open(f))
    n, b = d["numbers"]["head"], d["numbers"]["base"]
    net = d["run"].rsplit("_s", 1)[0]
    diff = {k: n[k] - b[k] for k in ("topo_F1", "F1@1.0", "dup_frac", "vert100_dp", "junction_F1", "lanes_found", "junction_R")}
    holds = diff["topo_F1"] > 0.02 and diff["F1@1.0"] >= -0.01 and diff["dup_frac"] <= 0 and diff["vert100_dp"] <= 1.0 and diff["junction_F1"] >= -0.03
    R[(net, d["thr"])][d["seed"]] = (holds, diff)
for (net, thr), v in sorted(R.items(), key=lambda x: (x[0][0], -x[0][1])):
    print(f"{net:10s} thr {thr}: " + " | ".join(f"s{s}: {'HOLDS' if h else 'fails'} topo {d['topo_F1']:+.4f} F1@1 {d['F1@1.0']:+.4f} dup {d['dup_frac']:+.4f} vert {d['vert100_dp']:+.2f} jF1 {d['junction_F1']:+.4f}" for s, (h, d) in sorted(v.items())))
print()
for thr in (0.5, 0.2, 0.1):
    cells = [R[(net, thr)].get(s) for net in ("e287_p2", "e305_e") for s in (0, 1)]
    done = [c for c in cells if c]
    print(f"thr {thr}: {len(done)}/4 confirmation cells scored; all hold: {all(c[0] for c in done) and len(done) == 4}")
