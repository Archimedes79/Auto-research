p = "/mnt/c/Auto-research/winners/README.md"
t = open(p, encoding="utf-8").read()
old = [l for l in t.split("\n") if l.startswith("| `e365_token_champion/`")][0]
new = "| `e365_token_champion/` | the end-to-end token champion (D113 / D114, E373): one token head on the predicted nodes of the frozen lane head B', trained with the none weight 0.25 (the runner-up, none weight 1.0, is the same code; source + tests; it needs the repository, see its README) | topo F1 0.6698 / 0.6769 on two fresh head seeds (runner-up 0.6617 / 0.6767, 0.6618 / 0.6733), whole path 16.6 - 16.8 ms; staged, B' frozen |"
open(p, "w", encoding="utf-8").write(t.replace(old, new))
print("ok")
