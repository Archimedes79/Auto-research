s = open("HANDOVER.md", encoding="utf-8").read()
a = s.index("4. **E299** time card v2"); b = s.index("5. Simplification:")
new = '''4. **E299** time card v2 (critic REVISE answered) + **step 0 DONE** (`evaluation/jtime.py`, CPU, 2588c101): logit-only flips are 27.2 % / 26.5 % of the true pairs lost between two keyframes (the rest are end misses a re-scoring layer cannot fix), the zero-parameter replay at the pre-registered weight 0.5 gives pair precision +0.011 / +0.011 at the delivered operating point (logit -2.2; +0.005 / +0.003 at logit 0) and snapped topo F1 +0.0005 / +0.0001: step 0 passes narrowly, the training arms (9 GPU h) are LOW priority, they run only when the nights are free after E292 / E325 / E324.
'''
s = s[:a] + new + s[b:]
open("HANDOVER.md", "w", encoding="utf-8").write(s); print("ok")
