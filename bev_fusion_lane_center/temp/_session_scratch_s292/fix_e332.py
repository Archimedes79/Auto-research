p = "experiments/E332_maps-head-without-auxiliary-maps/experiment.md"
s = open(p, encoding="utf-8").read()
a = "the kill-check script's E307 value is read at its own threshold; `jmaps_val.json` best-threshold numbers:"
assert s.count(a) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, "the kill check reads the keypoint F1 at the threshold 0.3 as the card's kill states; the `jmaps_val.json` best-threshold numbers are"))
