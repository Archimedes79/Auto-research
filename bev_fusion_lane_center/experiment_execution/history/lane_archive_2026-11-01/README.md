# Lane runs archive, 2026-11-01

C: was full (4.1 GB free). On Mirko's instruction ("condense into the most important experiments, results and code; controlled delete; lane-centre system only") the heavy files of the old run folders in `bev_fusion_lane_center/runs/` were **moved to D:, not deleted**; the permanent delete is Mirko's step (agents never delete for good).

| item | value |
|---|---|
| moved | 615 files, 19.44 GB, from 92 run folders (files >= 1 MiB only; small files such as `metrics.json`, `status.json`, logs stay in place) |
| archive | `D:\Auto-research\lane_archive_2026-11-01\runs\<same relative path>` |
| list of moved files | `MANIFEST.csv` (entry, relpath, bytes, reason) |
| kept on C: | 561 entries, 7.85 GB: `KEEP_ON_C.txt` (entry, size, reason) |
| verification | every manifest row exists in the archive with the same byte size and no file remains at the source: 615 / 615 (checked 2026-11-01 right after the move; robocopy /MOV copies, then removes the source file) |
| C: free | 4.1 GB before, 36.4 GB after (the move accounts for 19.4 GB of it) |

## Rule that decided
Stays on C: the registered systems (`models/registry.json`), the D7 lineage and the map-branch builders (`runs/e123/e123_D7_*`, `e17_*`, `e18_*`, `e24_*`, `e25_*`), `e123_ladder` (cards), the control `e94_a` (its stem initialises every builder), the inputs the live product chain reads (`e37_fold0/1/2`, `e37_final`, `e17/dense_trainoof`, `e57_stored`), the joint models behind the TEST check E230 (`e18_joint*`), the loop-18 workspace and anything written in the last five days or at most 20 MB.
Moved: everything else of 20 MB or more: the loop 1 - 15 runs (CRN, dense-head variants, failure sweeps e57 / e65 / e59p, route-head and token caches of old loops, rule-graph system A), the loop-17 intermediate arrays (`e17/dense_e37`, `dense_ftall`, `e139_*`, `e18_r21*`, ...). Their metrics, logs, cards and the experiment summaries stay, so old numbers can be quoted but no moved run can be re-evaluated from a checkpoint.

## Known leftovers
- Four registry names point to run folders that have no `model.pt` and no card (they were already like that before the move): `e24_j1phd_s0` (the run was cut at 3200 of 4800 iterations), `e24_j1phd_s1`, `e24_j1cl8_s1`, `e25_j1phd2_s1`. `evaluate.py` on those names fails; the entries could be removed from the registry in a decision of Mirko (registry is protected for champion / control only).
- Legacy drivers in `training/legacy/` and `training/data/prep/` still name moved folders in their defaults (`e60_main`, `e27_gcar`, `e32_main`, `e44_nodes`, `e37_ens`, `e113_a`, `e43`). They serve finished experiments E19 - E72 and fail unless the folder is restored.

## Restore
Everything back (moves the files home again):
```
robocopy D:\Auto-research\lane_archive_2026-11-01\runs C:\Auto-research\bev_fusion_lane_center\runs /E /MOV /R:1 /W:2
```
One folder, e.g. `e60_main`:
```
robocopy D:\Auto-research\lane_archive_2026-11-01\runs\e60_main C:\Auto-research\bev_fusion_lane_center\runs\e60_main /E /MOV
```

## Permanent delete (Mirko, when the condensed record and the kept set are accepted)
```
Remove-Item -Recurse -Force D:\Auto-research\lane_archive_2026-11-01
```
That frees 19.4 GB on D:. Nothing on C: depends on it.

## Condensed record
`reports/source/Condensed_Record_2026-11-01.md`: the key experiments (K1 - K26) with their numbers, what was dropped, sources.
