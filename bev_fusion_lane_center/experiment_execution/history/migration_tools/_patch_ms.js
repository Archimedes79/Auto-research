const fs=require("fs");
const p="C:/ML-Experiment/bev_fusion_lane_center/tools/maintenance/make_structure.py";
let s=fs.readFileSync(p,"utf8");
s=s.split("\n").filter(l=>!new RegExp("^    \"scripts[^\"]*\":").test(l)).join("\n");
const add=`    "training": "the system: models, data (+prep), losses, train, pipeline, search; trainer.py = general trainer with checkpoint + resume",
    "training/models": "network definitions (builder D1-D8, trace network E94, junction / piece heads, refinement E120)",
    "training/data": "datasets, constants, libraries; prep/ = builders of the cached data",
    "training/train": "training entry points (builder, route head, consumer, dense head, control arm)",
    "training/pipeline": "inference and assembly: path union, CUDA-graph runtime, chain builder -> route head -> consumer",
    "training/search": "successive-halving harness (p16_triage)",
    "evaluation/drivers": "entry points that run an evaluation (final eval, fast check, timing, route eval, bootstrap)",
    "evaluation/tests": "metric response test, builder spec tests",
    "evaluation/examples": "champion vs control on all validation frames: numbers, paired bootstrap, images, video",
    "temp": "short-lived scripts by loop (analysis, launchers, archive): delete when the loop is filed",
    "temp/analysis": "diagnostics and explorations of single loops",
    "temp/launchers": "shell wrappers of finished batches (record of the command lines)",
    "temp/archive": "historic scripts nothing imports",
    "reports/build": "report builders, figure scripts, PDF printing",
`;
s=s.replace('    "tools": ',add+'    "tools": ');
s=s.replace('"literature", "experiment_execution", "experiments", "evaluation", "reports", "scripts", "tools"','"literature", "experiment_execution", "experiments", "training", "evaluation", "tools", "reports", "temp"');
s=s.replace('"tools": "reusable helpers: viz (images), video, experiment (run.py, ledger.py), env (setup), maintenance, restructure (migration of 2026-10-24)"','"tools": "reusable helpers: env (launchers), experiment (run.py, ledger.py), viz (images), video, catalogue, maintenance"');
fs.writeFileSync(p,s);
