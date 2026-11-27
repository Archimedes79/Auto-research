# templates

What an agent copies when it starts a file of a problem.

| file | for | read by |
|---|---|---|
| `experiment.md` | one experiment: the `json card` block, the Design delta, the plan, the conclusion | `research` (`spec.py` checks the card) |
| `problem.md` | `PROBLEM.md` of a new problem (goal, chain, metrics, current best, gaps) | the owner, the planner |
| `paper_note.md` | one note per paper; its eight header lines become the topic table | the researcher (mode deep), `literature/topic_table.py` |
| `group_summary.md` | the overview of a topic or a problem group | the researcher (mode summary), for the planner |
