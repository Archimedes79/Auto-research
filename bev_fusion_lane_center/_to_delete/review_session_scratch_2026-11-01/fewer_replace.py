"""one-off: wording for the card block (fewer files); usage: fewer_replace.py <in> <out> <set>"""
import sys

SETS = {
    "docs": [
        ("`experiment.json` of every card the tools decide", "card block (in `experiment.md`) of every card the tools decide"),
        ("also the arms of an `experiment.json`", "also the arms of a card"),
        ("the `experiment.json` of every card the tools decide", "the card block (in `experiment.md`) of every card the tools decide"),
        ("applies the rule of `experiment.json`", "applies the rule of the card (the block in `experiment.md`)"),
        ("and, for runs the tools decide, `experiment.json` (", "and, for runs the tools decide, its card block (section 0; "),
        ("`experiment.json` per card the tools decide", "the card block per card the tools decide"),
        ("experiment.md (+ .json)", "experiment.md (card block)"),
    ],
    "readme": [
        ("(card -> detail plan -> summary, next to `experiment.json`)", "(card block -> detail plan -> summary, all in one file)"),
        ("rule clauses of experiments/E240_*/experiment.json on", "rule clauses of the card of experiments/E240_*/experiment.md on"),
        ("gate G2 as code for experiment.json (schema: schemas/experiment.schema.json)", "gate G2 as code for the card block of experiment.md (an old experiment.json is still read; `migrate` moves it; schema: schemas/experiment.schema.json)"),
    ],
}

src, dst, which = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
for old, new in SETS[which]:
    n = s.count(old)
    print(n, "x", old[:60])
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
