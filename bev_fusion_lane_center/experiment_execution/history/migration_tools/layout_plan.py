"""Plan of the role-based code layout:  training/ (models, data, losses, train, pipeline, prep)  evaluation/ (metrics + drivers)  tools/ (few major tools)  reports/build/  temp/ (short-lived).
Dry run: prints the numbers and the core list;  writes restructure/layout.json.   Run: _run.sh layout_plan.py"""
import ast
import json
import os
import re
from collections import defaultdict

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
SC = f"{R}/scripts"
files = sorted(f for f in os.listdir(SC) if os.path.isfile(f"{SC}/{f}"))
mods = {f[:-3] for f in files if f.endswith(".py")}
extra_dirs = [f"{R}/evaluation/lane_grid", f"{R}/evaluation/sparse_lanes", f"{R}/tools/viz", f"{R}/tools/video"]
ext_mods = set()
for d in extra_dirs:
    ext_mods |= {f[:-3] for f in os.listdir(d) if f.endswith(".py")}
imports = {}
for f in files:
    if not f.endswith(".py"):
        continue
    try:
        tree = ast.parse(open(f"{SC}/{f}", encoding="utf-8-sig", errors="replace").read())
    except SyntaxError:
        imports[f[:-3]] = set()
        continue
    s = set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            s.update(a.name.split(".")[0] for a in nd.names)
        elif isinstance(nd, ast.ImportFrom) and nd.module:
            s.add(nd.module.split(".")[0])
    imports[f[:-3]] = s & mods

CORE_ROOTS = ["p16_fast_check", "p16_fast_val", "p16_pipeline_eval", "p16_pipeline", "p16_train", "p16_triage", "p16_ctrl", "p16_timing", "p16_pareto", "p16_final_eval", "p16_eval", "p16_runtime",
              "p10_route_train", "p10_route_eval2", "p14_chain_consumer", "p14_chain_data", "p16_chain_data", "p16_tok_resample", "p16_route_check", "p15_refine", "p16_ensemble", "p16_hyst", "dense_head",
              "graph_from_dense", "test_p16_spec", "boot_eval", "p11_eval", "p17_latency", "p10_train", "p10_infer", "p16_infer", "p16_junc_head_eval", "p16_diag", "oracle_vertices", "p12_calib"]
core, todo = set(), [m for m in CORE_ROOTS if m in mods]
while todo:
    m = todo.pop()
    if m in core or m not in mods:
        continue
    core.add(m)
    todo.extend(imports.get(m, []))

REPORTS = re.compile(r"^(build_|loop1[0-9]_text|fig_|arch_figs|report_lib|finish_report|dump_catalogue|lit16_merge|_montage_pptx|_readme16|_regen_rung_figs|_pdfpages|p16_report|p16_catalogue|e2e_catalogue|e2e_basis)")
CATALOGUE = re.compile(r"^(e2e_catalogue|e2e_basis|build_experiment_table|p16_catalogue|dump_catalogue_json|lit16_merge)")
RUNNER = re.compile(r"^(_py\.sh|_py14\.sh|_pyr\.sh)$")


def dest(f):
    b = f[:-3] if f.endswith(".py") else f
    if f == "README.md":
        return None
    if RUNNER.match(f):
        return "tools/env"
    if CATALOGUE.match(b):
        return "tools/catalogue"
    if REPORTS.match(f):
        return "reports/build"
    if b in ("graph_eval",):
        return None
    if f.endswith(".py") and b in core:
        if b in ("p16_loss",):
            return "training/losses"
        if re.match(r"^(p16_(builder|gate|geom|tokens|time|anchors|ref)|p10_(net|expert|wave)|p13_(jhead|junction|phead|connect|consumer|consumer_ego)|p8_(assign|consumer|seg)|p15_refine|line_tracer|route_head)", b):
            return "training/models"
        if re.match(r"^(p16_data|p10_data|p16_chain_data|p14_chain_data|p14_chain_fail_data|p15_chain_data|p15_make_tokens|p8_lib|sd_lib|assoc_lib|centerline_gt|gcar_tools)", b):
            return "training/data"
        if re.match(r"^(p16_train|p10_train|p10_route_train|p14_chain_consumer|p14_chain_fail_train|p14_joint|p15_joint|dense_head)", b):
            return "training/train"
        if re.match(r"^(p16_(infer|runtime|pipeline|ctrl|ensemble|hyst|tok_resample|common|asm_grid|driver)|p10_infer|graph_from_dense)", b):
            return "training/pipeline"
        if re.match(r"^(boot_eval|p11_eval|p12_calib|oracle_vertices|p10_route_eval2|p16_(eval|final_eval|fast_check|fast_val|pipeline_eval|timing|pareto|route_check|junc_head_eval|diag|triage)|p17_latency|test_p16_spec)", b):
            return "evaluation/drivers"
        return "training/pipeline"
    return "temp"


plan = {}
for f in files:
    d = dest(f)
    if d:
        plan[f] = d
cnt = defaultdict(list)
for f, d in plan.items():
    cnt[d].append(f)
for d in sorted(cnt):
    print(f"{d:24s} {len(cnt[d]):4d}")
print()
for d in ("training/models", "training/data", "training/losses", "training/train", "training/pipeline", "evaluation/drivers", "tools/catalogue", "tools/env", "reports/build"):
    print(f"[{d}]", " ".join(sorted(cnt[d]))[:1800])
print("core modules (closure):", len(core))
extra_core = sorted(m for m in core if plan.get(m + ".py") == "temp")
print("core but landing in temp (bug if any):", extra_core)
json.dump(plan, open("/mnt/c/ML-Experiment/restructure/layout.json", "w"), indent=0)
