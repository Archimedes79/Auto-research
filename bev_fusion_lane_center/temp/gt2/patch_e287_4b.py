p = "experiments/E287_junction-pair-head-v2/experiment.md"
s = open(p, encoding="utf-8").read()
s = s.replace("the head delivers 67 - 70 % of what its keypoints allow, the rule 83 - 78 %.", "the head delivers 60 - 70 % of what its keypoints allow, the rule 78 - 83 %.")
s = s.replace("the pair decision still leaves 0.13 of the pairs on the table at logit >= 0", "the pair decision still leaves 0.12 - 0.20 of the pairs on the table at logit >= 0")
open(p, "w", encoding="utf-8").write(s)
