# agents

The four agent roles of the method (`../METHOD.md`, "Roles"). Each file is a short, stable English role text with a frontmatter; `../tools/install_agents.py` lints them (no paths, names, dates or run ids) and copies them to the workspace's `.claude/agents/`, where Claude Code finds them (`--check` tells when the copies drift). This README is not copied.

| file | role | modes |
|---|---|---|
| `ar-planner.md` | the planner | problem, branches (one lens per run), plan, detail, tracking |
| `ar-researcher.md` | the researcher | scan, deep, check, summary |
| `ar-coder.md` | the coder of the inner loop: a coding step with its spec test | - |
| `ar-critic.md` | fresh eyes on what it did not write | cards, code, numbers |
