def sub(p, a, b, new=".new_s292"):
    s = open(p, encoding="utf-8").read()
    assert s.count(a) == 1, (p, a[:60])
    open(p + new, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
sub("tools/maintenance/run_gates.sh", 'run "head: previous-keyframe ceiling and replay (E299 step 0)"  jtime.py selftest\n',
    'run "head: previous-keyframe ceiling and replay (E299 step 0)"  jtime.py selftest\nrun "head: wiring twin of the nearest-piece rule, kept ends (E325)"  grid/jwire.py\n')
p = "training/grid/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("| `test_mnms_layer.py` |"))
L[i + 1:i + 1] = ["| `jwire.py` | E325 (step 1): the wiring of a connector end to a delivered piece end as replaceable functions: `kept_ends` (the pieces that survive the long-link cover drop of `qends_eval.add_connectors` and their ends, the one place the read-out and a head's features compute it), `candidates` (reach 4 m, 45 degrees, at most 8 places), `twin_pick` / `twin_choice` (the zero-parameter twin: argmax of softmax(-d^2 / (2 sigma^2)) = the nearest-piece rule, ties in the list order); `add_connectors(wire_choice=...)` takes one (entry index, exit index) per pair; the head itself is a later step of the card | new | `python jwire.py` (twin against the rule on 1200 ends, through `add_connectors` on 300 random graphs incl. covered long links, mutations: a swapped choice changes the graph, a wrong length / no snap / an index outside the kept pieces raise); `jends_eval.py score --snap --wire twin --tag twin` reproduces the E312 cards |"]
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write("\n".join(L))
p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
a = "(`card_name`, `score_target`; the card carries `full`, `spec`, `tag`)"
assert s.count(a) == 1
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s.replace(a, a + "; E325: `--wire twin` (with `--snap` and a tag) lets `jwire.twin_choice` choose the wired pieces through `add_connectors(wire_choice=...)`: it must reproduce the card of record"))
print("ok")
