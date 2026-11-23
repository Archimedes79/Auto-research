# winner - the winning system of the lane project and its code

One folder per winner (D139 / D140, 2026-11-23: every project keeps its winner with the code that belongs to it; until then the folders lived in a workspace folder named winners). A winner folder is small and self-describing: the **source code** of the system, a README with the exact commands and the numbers to reproduce, and a manifest (`e372_s4_champion/MANIFEST.json`); **no weights, no data, no logs** (the weights stay in `../runs/`, named in the winner's README).

| folder | what | numbers (VAL 873 frames, seed 0 / seed 1) |
|---|---|---|
| `e372_s4_champion/` | the end-to-end token champion since D120 (E372 S4-HOLDS): the stage-1 head of E373 on the frozen lane head B', then the head and B''s node head trained together through the token loss (same inference network, 2.323 M parameters, 16.21 ms quiet) | topo F1 0.6682 / 0.6813 at the D115 decode |

## Make and check one (tool: `../../reporter/winner.py`)
```
python reporter/winner.py export <name> --out bev_fusion_lane_center/winner --src "<folder>/**/*.py" --strip <folder> --numbers "..." --command "..." --weights "..."
python reporter/winner.py verify bev_fusion_lane_center/winner/<name>      # the folder against its manifest; reports drift against the sources
```
`export` refuses weight-like files, any file over 300 KB, a folder over 1 MB and sources with uncommitted changes, so a winner folder always matches a commit.

## Rules
Only learned systems of the solution set (D87 / D109 / D113: end-to-end learned) are winners; a system with rule steps is not (the former winner folder of E324 s is kept in the local git tag `archive/winner-e324s-lanegraph-2026-11-19`; the stage-1 folder `e365_token_champion` is in the git history, commit `57b9eab1`). A winner folder is a snapshot: the code of record stays where it is developed (`../training/`, `../evaluation/`); a winner is exported when the system has won by a written rule and its tests are green, and exported again when the champion changes. `verify` tells when the code of record has drifted from the snapshot (the E392 / E393 options added since 2026-11-18 are off by default and leave the champion's training bit for bit, checked on e392_s4_c_s0).
