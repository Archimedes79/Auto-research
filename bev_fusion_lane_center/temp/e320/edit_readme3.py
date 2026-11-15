head = open("temp/e320/README_eval_head.md", encoding="utf-8").read().split("\n")
work = open("evaluation/README.md", encoding="utf-8").read().split("\n")
jt = [l for l in work if l.startswith("| `jtest_eval.py` |")]
assert len(jt) == 1 and not any(l.startswith("| `jtest_eval.py` |") for l in head)
out = []
for l in head:
    out.append(l)
    if l.startswith("| `jmaneuver.py` |"):
        out.append(jt[0])                                                       # the row that HEAD lacks (the working copy has it)


def append_to(prefix, old_tail, new_tail):
    for i, l in enumerate(out):
        if l.startswith(prefix):
            assert l.rstrip().endswith(old_tail), (prefix, l[-80:])
            out[i] = l.rstrip()[: -len(old_tail)] + new_tail
            return
    raise AssertionError(prefix)


append_to("| `qends_eval.py` |", "|", "(`add_connectors(wire_pos=r)`: under a tight radius the cover radius of B's long links follows it, min(4 m, r)) |")
append_to("| `jmaneuver.py` |", "|", "`parse_variants` is the one place that reads the variant names (tested) |")
append_to("| `jdiag.py` |", "|", "`attlog` = the attachment statistics of an E320 training log, the mean of the last 10 rows weighted by att_n (the kill criterion of the card) |")
open("temp/e320/README_eval_blob.md", "w", encoding="utf-8").write("\n".join(out))
for k in ("jmodel_cost.py", "jrule_removal.py", "jtest_eval.py", "p0.1an", "boot", "make_pairs", "attlog", "parse_variants", "attach-r"):
    assert any(k in l for l in out), k
print("readme blob ok", len(out), "lines")
