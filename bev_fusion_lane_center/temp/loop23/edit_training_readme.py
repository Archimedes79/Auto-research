p = "training/README.md"
L = open(p, encoding="utf-8").read().split("\n")
for i, ln in enumerate(L):
    if ln.startswith("| `trainer.py` |"):
        assert ln.rstrip().endswith("|")
        L[i] = ln.rstrip()[:-1].rstrip() + (" The run directory follows the trainsuite contract (2026-11-03, copied from the BEV system's trainer): `config.json`, `log.jsonl` (rows with total, lr, grad_norm, "
                                              "s_per_step, gpu_gb, the loss's scalars; `probe=True` adds `grad.<module>`, `upd.<module>`, `t_data`, `t_step`), `train.log`, a stop request "
                                              "(`python -m trainsuite stop RUN`: checkpoint, `STOPPED`, exit code 75), so `python -m trainsuite status / health / curves / watch` and `run.py` work on it. |")
        L.insert(i + 1, "| `probe.py` | gradient and update size per module, data wait against compute time, `incident_<step>.json` dump (copied unchanged from `bev_fusion/training/probe.py`; it knows no project code) | used by `trainer.fit(probe=True)` |")
        break
else:
    raise SystemExit("trainer row not found")
s = "\n".join(L)
a = "- Everything that trains must be resumable and must write `log.txt`, `status.json`, `model.pt` into `runs/<name>/` (use `trainer.fit`;"
assert s.count(a) == 1
s = s.replace(a, "- Everything that trains must be resumable and must write `log.txt` / `train.log`, `log.jsonl`, `config.json`, `status.json`, `model.pt` into `runs/<name>/` (use `trainer.fit`: that is the trainsuite contract;")
open(p, "w", encoding="utf-8").write(s)
print("ok")
