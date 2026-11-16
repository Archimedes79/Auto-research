import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, s))
import jnarrow as JN
import jprune_probe as PP
import jprune_cost as PC
cache = PP.build_cache("val", workers=0, max_frames=16)
xs = [b["x"].float() for b in cache]
out = {}
for S in (0, 1):
    run = f"e324_s_s{S}"
    spec = {"calib": f"temp/e345/e324_s_s{S}/calib.pt", "edits": [{"unit": "jm.m05.hid", "frac": 0.75}]}
    full, args = JN.JEV.load_model(run, "cpu")
    nar, _ = JN.build_narrowed(run, spec)
    sp = nar.narrow_spec
    es = JN.expected_saving(full, sp)
    mf, mn = JN.measured_macs(full), JN.measured_macs(nar)
    npar = lambda m: sum(p.numel() for p in m.parameters())
    d = JN.difference(full, nar, xs)
    t = sp["taps"]["jm.m05.2"]
    kept_groups = [g for g in range(8) if g * 4 not in t["removed"]]
    # probe's own per-channel point (E345 s3 anchor) for comparison of the sets
    calib = PP.load_calib(PP.rootpath(spec["calib"]), "cpu")
    probe = PP.spec_edits({"jm.m05.hid": 0.75}, PP.build_units(full), calib)[0].idx
    groups_of_probe_kept = sorted({c // 4 for c in range(32) if c not in probe})
    out[S] = dict(kept_groups=kept_groups, removed_channels=len(t["removed"]), summed_importance_per_group=[round(float(x), 4) for x in calib.s["jm.m05.2"].view(8, 4).sum(1)],
                  probe_kept_groups_touched=groups_of_probe_kept, cost_saved_gmacs=es["macs"] / 1e9, cost_saved_params=es["params"], measured_saved_gmacs=mf["total"] - mn["total"],
                  measured_maps=[mf["maps"], mn["maps"]], measured_total=[mf["total"], mn["total"]], params=[npar(full), npar(nar)], diff=d, sha=sp["calib_sha256"])
    print(S, json.dumps(out[S], indent=1), flush=True)
json.dump(out, open("temp/e353/numbers_n1.json", "w"), indent=1)
