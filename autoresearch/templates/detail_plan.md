# Detail plan - E<id> <short title> (card H<..>)

**Understanding of the card** (in our own words; what would make it fail): ...

## Steps
| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | ... | ... | spec test `<name>` passes; gradient reaches the new parameters | |
| train | config / arms / seeds | `run.py --id E<id>.<arm> ...` | done flag, checkpoint | |
| evaluate | registry names, splits, `--compare` | evaluation entry point | card written | |
| decide | rule of the card | `decide.py ... --write` | verdict in the ledger | |
| summarise | table, images, decision, open points | experiment folder | summary written | |

Ledger rows (planned): E<id>.a, E<id>.c ...
Schedule items: see `schedule.csv`
