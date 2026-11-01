p = "CLAUDE.md"
s = open(p, encoding="utf-8").read()
old1 = "One git repository holds the code and text of the first three (whitelist in `.gitignore`); `datasets/` and `wsl/` are not in git."
new1 = "One git repository holds the code and text of all four (whitelist in `.gitignore`; of `datasets/NuScenes/` only the README and `tools/`, never data); `wsl/` is not in git."
old2 = "not yet managed by `autoresearch/` (no project.json, ledger or run.py: runs are launched by `temp/train/run_cfg.sh`); duplicating code from the lane project is allowed."
new2 = "managed by `autoresearch/` since D23 (2026-10-31: `bev_fusion/project.json`, tag prefix `bf`; trainings start through `run.py`, which calls `temp/train/run_cfg.sh` with the run's tag - the start line is in `bev_fusion/records/README.md`; the tools need `AR_PROJECT=../bev_fusion`, the default problem is the lane one); duplicating code from the lane project is allowed."
old3 = "shared by all problems (not in git)."
new3 = "shared by all problems (README and tools in git, the data never)."
for o, n in ((old1, new1), (old2, new2), (old3, new3)):
    assert s.count(o) == 1, o[:40]
    s = s.replace(o, n)
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
