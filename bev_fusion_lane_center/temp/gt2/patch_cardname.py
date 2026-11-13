p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = '''    name = "ends_card.json" if (not snap and a.thr == 0.5) else "ends_card_" + ("snap" + "".join("_" + k[5:] for k in opts) + "_" if snap else "") + f"thr{a.thr}.json"'''
assert old in s
new = '''    chord = bool(getattr(a, "chord", False))
    name = "ends_card.json" if (not snap and not chord and a.thr == 0.5) else "ends_card_" + ("chord_" if chord else "") + ("snap" + "".join("_" + k[5:] for k in opts) + "_" if snap else "") + f"thr{a.thr}.json"     # the chord control has its own file too (it overwrote the card of the learned scales on 2026-11-12)'''
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
