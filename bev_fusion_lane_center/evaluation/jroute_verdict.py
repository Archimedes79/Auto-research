"""E389b: the arithmetic of the card's rule (by hand: the card's section 0 / 'The rule') from the stored score cards -> runs/eval/e389b_verdict.json. stdlib only. usage (project folder): python3 evaluation/jroute_verdict.py"""
import glob
import json

R = "runs"
BAND = 0.010


def K(path):
    return json.load(open(path))["by_delivered"]["kept"]


def one(pattern):
    fs = glob.glob(pattern)
    assert len(fs) == 1, (pattern, fs)
    return fs[0]


out = dict(rule="card E389b (ecf42b8811), by hand; arithmetic by evaluation/jroute_verdict.py", band=BAND, seeds={})
ok1 = ok2 = ok3 = True
for s in (0, 1):
    z = K(f"{R}/e389_z_s{s}/route_card_error.json")["numbers"]
    row = dict(z=dict(error=z["route_F1_road"], fork=z["fork_F1_road"], lane=z["route_F1_lane"]))
    for tag, run in (("m", f"e389_m_s{s}"), ("a", f"e389b_a_s{s}")):
        d = dict(error=K(one(f"{R}/{run}/route_card_error_vs_e389_z_s{s}.json")))
        n = d["error"]["numbers"]
        res = dict(error=n["route_F1_road"], fork=n["fork_F1_road"], lane=n["route_F1_lane"], exact=None, none=None, wrong=None, gross=None)
        for mode in ("exact", "none", "wrong", "gross"):
            res[mode] = K(one(f"{R}/{run}/route_card_{mode}_vs_e389_z_s{s}_rmerror.json"))["numbers"]["route_F1_road"]
        res["error_minus_z"] = res["error"] - z["route_F1_road"]
        res["fork_minus_z"] = res["fork"] - z["fork_F1_road"]
        res["lane_minus_z"] = res["lane"] - z["route_F1_lane"]
        res["none_minus_z"] = res["none"] - z["route_F1_road"]
        res["wrong_minus_z"] = res["wrong"] - z["route_F1_road"]
        res["interval_error_minus_z"] = d["error"]["paired"]["route_F1_road"]
        row[tag] = res
    pa = K(one(f"{R}/e389b_a_s{s}/route_card_error_vs_e389_m_s{s}.json"))["paired"]
    row["a_minus_m"] = {k: pa[k] for k in ("route_F1_road", "route_F1_lane", "fork_F1_road")}
    m, a = row["m"], row["a"]
    row["c1_road"] = a["error_minus_z"] >= m["error_minus_z"] - BAND
    row["c2_none"] = a["none_minus_z"] >= m["none_minus_z"] - BAND
    row["c2_wrong"] = a["wrong_minus_z"] >= m["wrong_minus_z"] - BAND
    row["c3_fork"] = a["fork_minus_z"] >= m["fork_minus_z"] - BAND
    ok1 &= row["c1_road"]
    ok2 &= row["c2_none"] and row["c2_wrong"]
    ok3 &= row["c3_fork"]
    out["seeds"][str(s)] = row
p = K(one(f"{R}/e389b_p_s0/route_card_error_vs_e389_z_s0.json"))["numbers"]
z0 = out["seeds"]["0"]["z"]
out["plain_s0"] = dict(error=p["route_F1_road"], error_minus_z=p["route_F1_road"] - z0["error"], fork=p["fork_F1_road"], fork_minus_z=p["fork_F1_road"] - z0["fork"], lane=p["route_F1_lane"],
                       a_minus_p=out["seeds"]["0"]["a"]["error"] - p["route_F1_road"])
for mode in ("exact", "none", "wrong", "gross"):
    out["plain_s0"][mode] = K(one(f"{R}/e389b_p_s0/route_card_{mode}_vs_e389_z_s0_rmerror.json"))["numbers"]["route_F1_road"]
kill = {}
for run in ("e389b_a_s0", "e389b_a_s1", "e389b_p_s0"):
    L = [json.loads(l) for l in open(f"{R}/{run}/log.jsonl") if l.strip()]
    r = {x["step"]: x["total"] for x in L if "step" in x and "total" in x}
    a0 = [r[k] for k in r if k <= 100]
    a1 = [r[k] for k in r if 2900 < k <= 3000]
    kill[run] = sum(a1) / len(a1) / (sum(a0) / len(a0))
out["loss_ratio"] = kill
out["kill_loss_fired"] = any(v >= 0.8 for v in kill.values())
out["kill_ignores_map_a"] = all(abs(out["seeds"][str(s)]["a"]["wrong"] - out["seeds"][str(s)]["a"]["error"]) < 0.01 for s in (0, 1)) and all(out["seeds"][str(s)]["a"]["error_minus_z"] < 0.01 for s in (0, 1))
out["verdict"] = "ONE ATTENTION SUFFICES" if (ok1 and ok2 and ok3 and not out["kill_loss_fired"] and not out["kill_ignores_map_a"]) else "see clauses"
out["prediction"] = dict(value=0.09, mean_a_minus_z=sum(out["seeds"][str(s)]["a"]["error_minus_z"] for s in (0, 1)) / 2)
out["prediction"]["hit"] = abs(out["prediction"]["mean_a_minus_z"] - 0.09) <= 0.02
out["params"] = dict(ab=33738, a=33226, plain=32806)
json.dump(out, open("runs/eval/e389b_verdict.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1))
for s in ("0", "1"):
    r = out["seeds"][s]
    print(s, {k: round(r[tag][k], 4) for tag in ("m", "a") for k in ("error_minus_z",) for _ in (0,)}, [r[c] for c in ("c1_road", "c2_none", "c2_wrong", "c3_fork")])
