import sys
sys.path.insert(0, ".")
import evaluation.bev_gtcheck as G
import evaluation.bev_suite as S
task = G.build_task()
samples = G.pick(6, G.resolve_sig(None), 0)
outs = G.variants(samples, task.spec, task.groups, 0)["oracle"]
res = G.run(samples, outs, task)                      # keys "<metric>/<rest>"
table = {k.replace("/", ".", 1): v for k, v in res.items()}
flat = S.to_legacy(table)
better = S.better_of(table, task.metrics())
print(len(flat), "flat numbers;", sum(k.startswith(("detf/", "detfp/", "cov/")) for k in flat), "of the fusion family;", sum(k.startswith(("detf/", "detfp/", "cov/")) for k in better), "with a direction")
print([k for k in flat if k.startswith("cov/") and "vehicle" in k][:4])
print(S.headline_table(flat).split("\n")[-14:])
assert any(k.startswith("detf/") and k in better for k in flat) and "cov/dt+0.00/vehicle/care/recall2m" in better and "cov/dt+0.00/vehicle/care/n_gt" not in better
assert better["cov/dt+0.00/vehicle/care/ate"] == "lower" and better["cov/dt+0.00/vehicle/care/recall2m"] == "higher"
print("names ok")
