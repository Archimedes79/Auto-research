# winners

Small, self-describing folders with the **source code of winning systems**, one folder per winner, made to be handed over and to live in git. No weights, no data, no logs.

| folder | what | numbers (VAL, seed 0 / seed 1) |
|---|---|---|
| `e324s_lanegraph/` | E324 s, the end-to-end lane graph from camera + radar features (inference package; the previous winner, five hard steps) | topo F1 0.6912 / 0.6948 at the end-to-end read-out |
| `e365_token_champion/` | stage 1 of the token champion, kept as the history of before D120 (D113 / D114, E373): one token head on the predicted nodes of the frozen lane head B', trained with the none weight 0.25 (the runner-up, none weight 1.0, is the same code; source + tests; it needs the repository, see its README) | topo F1 0.6698 / 0.6769 on two fresh head seeds (runner-up 0.6617 / 0.6767, 0.6618 / 0.6733), whole path 16.6 - 16.8 ms; staged, B' frozen |
| `e372_s4_champion/` | the end-to-end token champion since D120 (E372 S4-HOLDS): the stage-1 head of E373 on the frozen lane head B', then the head and B's node head trained together through the token loss (two stages; same inference network, 2.323 M parameters, 16.21 ms quiet) | topo F1 0.6682 / 0.6813 at the D115 decode |

## One winner folder
`code/` (the files, relative to their source folder), `README.md` (what it is, the exact command, the numbers to reproduce, where the weights are, the source commit), `MANIFEST.json` (every file with its sha256 and source path, the git commit, whether the sources had uncommitted changes).

## Make and check one (tool: `reporter/winner.py`)
```
python reporter/winner.py export <name> --src "<folder>/**/*.py" --strip <folder> --numbers "..." --command "..." --weights "..."
python reporter/winner.py verify winners/<name>      # the folder against its manifest; reports drift against the sources
```
`export` refuses weight-like files, any file over 300 KB, a folder over 1 MB and sources with uncommitted changes, so a winner folder always matches a commit.

## Rules
A winner folder is a snapshot: the code of record stays where it is developed; a winner is exported when the system has won by a written rule and its tests are green. Re-export after a change of the source; `verify` tells when a folder has drifted. The weights stay in the repository runs (named in the README of the winner).
