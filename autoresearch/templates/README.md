# templates

What agents copy when they start a file of a problem. The tools read some of them, so their markers and headers are a contract.

| file | for | read by |
|---|---|---|
| `problem.md` | `PROBLEM.md` of a new problem (goal, chain, metrics, current best, gaps) | the person, the planner |
| `experiment.md` | one experiment: the `json card` block (section 0), the detail plan, the summary | `tools/experiment.py`, `decide.py` (schema `../schemas/experiment.schema.json`) |
| `results.md` | the page of a decided experiment | `tools/result.py` writes between its markers |
| `plan.md` | an optional plan file of a loop | `tools/plan.py status` |
| `paper_note.md` | one research note; its eight header lines become the topic table | `tools/topic_table.py` |
| `group_summary.md` | the summary of a problem group, with its digest | the planner |
