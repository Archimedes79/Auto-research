p = "evaluation/README.md"
L = open(p, encoding="utf-8").read().split("\n")
for key, add in (("| `qends_eval.py` |", "E292: `score --full <dump of another head> --base <its checkpoint> --spec <read-out> --tag <name>` reads the delivered graph from the full dump of ANOTHER head (B'); the tag names the card (`ends_card_<tag>_...`), is mandatory with those options, and a tagged card is never overwritten (`card_name`, `score_target`; the card carries `full`, `spec`, `tag`)"),
                 ("| `jends_eval.py` |", "E292: `score` takes the same `--full --base --spec --tag` (the pyramid of the maps run must be the pyramid of `--base`: asserted)")):
    i = next(k for k, l in enumerate(L) if l.startswith(key))
    assert L[i].endswith(" |")
    assert "E292" not in L[i]
    L[i] = L[i][:-2].rstrip() + "; " + add + " |"
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("ok")
