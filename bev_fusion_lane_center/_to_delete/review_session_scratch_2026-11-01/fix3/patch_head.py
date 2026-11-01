"""apply only this session's direction edits to the HEAD versions of decide.py / experiment.py / selftest.py (head/ -> mine/)"""
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
WT = sys.argv[1]  # the working-tree tools folder: the new text of each hunk is copied from there


def wt(name):
    return open(os.path.join(WT, name), encoding="utf-8").read()


def between(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


edits = {
    "decide.py": [
        ("block", "# TEMPORARY (D24, 2026-10-31)", "\n\ndef card_lower(", "def lower_names(root=None):", "\n\ndef card_lower("),
        ("str", "    if undeclared and low != (metric in lower_names(root)):", "    if undeclared and low != named_lower(metric, root):"),
        ("str", "    problem's names (lower_names: project.json, the generic LOWER, the temporary lane list) for cards written before the field existed\"\"\"",
         "    problem's names (named_lower: project.json, the generic LOWER) for cards written before the field existed\"\"\""),
        ("str", "is undeclared and is read by the fallback lower_names.", "is undeclared and is read by the fallback named_lower."),
        ("str", "    return declared if declared is not None else metric in lower_names(root)",
         "    return declared if declared is not None else named_lower(metric, root)"),
        ("str", "    for older cards the generic cost names (LOWER) plus the problem's project.json \"lower_is_better\" list\n",
         "    for older cards the generic cost names (LOWER) plus the problem's project.json \"lower_is_better\" list (a path-like card key such as\n"
         "    'det/dt+0.00/vehicle/all/ate' is matched by its last part)\n"),
    ],
    "experiment.py": [
        ("str", "from decide import card_lower, lower_names  # noqa: E402", "from decide import card_lower, named_lower  # noqa: E402"),
        ("str", "        elif said is None and metric in lower_names(root) and exp[\"direction\"] != \"lower\":",
         "        elif said is None and named_lower(metric, root) and exp[\"direction\"] != \"lower\":"),
    ],
    "selftest.py": [
        ("block", "    # lower-is-better names: project.json", "    # the evaluator declares each metric's direction",
         "    # lower-is-better names (the fallback", "    # the evaluator declares each metric's direction"),
    ],
}
os.makedirs(os.path.join(D, "mine"), exist_ok=True)
for name, eds in edits.items():
    t = open(os.path.join(D, "head", name), encoding="utf-8").read()
    for e in eds:
        if e[0] == "str":
            assert t.count(e[1]) == 1, (name, e[1][:60], t.count(e[1]))
            t = t.replace(e[1], e[2])
        else:
            old = between(t, e[1], e[2])
            new = between(wt(name), e[3], e[4])
            assert t.count(old) == 1, (name, e[1])
            t = t.replace(old, new)
    open(os.path.join(D, "mine", name), "w", encoding="utf-8", newline="\n").write(t)
    print("patched", name)
