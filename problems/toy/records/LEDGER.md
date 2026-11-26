# Run ledger

Generated from the csv by `autoresearch/tools/ledger.py render` - edit the csv (or use `ledger.py note`), not this file.

## Loops

From `autoresearch/tools/metrics.py` (columns explained there; `-v` shows how each row was counted).

| loop | runs | failed | GPU h | exps | KEEP | CUT | TIE | INC | other | KEEP x2 | pred hit | mean err | TEST |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Runs

| id | status | hypothesis | prediction | rule | basis | seed | result | decision | started | min | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| toy.smoke_s0 | ok |  |  |  |  |  |  |  | 2026-11-26T22:15 | 0 | skipped |
| toy.dry_s0 | ok |  |  |  |  |  |  |  | 2026-11-26T22:16 | 0 | skipped |
| E1.c_s0 | ok | does a wider network segment the disc better (width=4) | +0.020 iou over arm c | [{"metric": "iou", "min_effect": 0.01}, {"metric": "recall", "max_loss": 0.02}] |  |  |  |  | 2026-11-26T22:17 | 0 | 1ae81e113d |
| E1.c_s1 | ok | does a wider network segment the disc better (width=4) | +0.020 iou over arm c | [{"metric": "iou", "min_effect": 0.01}, {"metric": "recall", "max_loss": 0.02}] |  |  |  |  | 2026-11-26T22:17 | 0 | c70bebed64 |
| E1.a_s0 | ok | does a wider network segment the disc better (width=12) | +0.020 iou over arm c | [{"metric": "iou", "min_effect": 0.01}, {"metric": "recall", "max_loss": 0.02}] |  |  |  | decide.py 2026-11-26: KEEP (effect +0.5937) | 2026-11-26T22:17 | 0 | b51319f9ec |
| E1.a_s1 | ok | does a wider network segment the disc better (width=12) | +0.020 iou over arm c | [{"metric": "iou", "min_effect": 0.01}, {"metric": "recall", "max_loss": 0.02}] |  |  |  | decide.py 2026-11-26: KEEP (effect +0.5937) | 2026-11-26T22:18 | 0 | 4d5d59d5a2 |
