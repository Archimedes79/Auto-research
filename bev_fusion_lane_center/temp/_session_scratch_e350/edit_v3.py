p = "evaluation/jwire_verdict.py"
s = open(p, encoding="utf-8").read()
old = '''        assert "no pair_run" in why({s: dict(good, wa=agree(pair_run=NONE)), 1 - s: good})
'''
new = '''        assert "no pair_run" in why({s: dict(good, wa=agree(pair_run=NONE)), 1 - s: good})
        swapped = agree()
        swapped["heads"] = swapped["heads"][::-1]                                                          # the first head must be head 0, not the ablation
        assert tok({s: dict(good, wa=swapped), 1 - s: good}) == "INVALID"
'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
p = "temp/_session_scratch_e350/mutate_tools.py"
s = open(p, encoding="utf-8").read()
old = '''("V22 the model_h suffix of heads 1 and 2 not required", V, '+ (f":model_h{h}.pt" if h else ""))', '+ "")'),'''
new = '''("V22 the model_h suffix of heads 1 and 2 not required", V, 'f"{a.head_prefix}{s}" + (f":model_h{h}.pt" if h else ""))', 'f"{a.head_prefix}{s}" + "")'),'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
