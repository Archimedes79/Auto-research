import io
p = "training/README.md"
s = io.open(p, encoding="utf-8").read()
anchor = "| `train/jtok_train.py` |"
row = ("| `data/lane_tokcache.py` | D113: the predicted-node cache: the tokens of a FROZEN lane head B (a `QGraph` run: B' = `runs/e292_a_s<S>`, plain B = `runs/e273_p_s<S>`) on every TRAIN and VAL frame (frames of `TokData(split, res=2.0)`), "
       "the K best cells by existence logit (stable order) with position, offset, heading, existence / flag / section / long-link logits and probabilities, the 64-d node token and the 144-d 2 m pyramid feature (float16), and the GT labels of every token "
       "(unique closest-first assignment within r = 1.5 m, type, flags, the GT links on tokens, the ceilings at K = 128 .. 512); one directory per run and split in `runs/tokcache/` (derived data, not in git); `build` (GPU, resumable), `fixture` "
       "(the cache against B's stored dump bit for bit), `report`, `selftest`; `TokCache` reads it, `nodes_of` gives the `qformat` nodes | `lane_tokcache.py selftest` |\n")
assert s.count(anchor) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(anchor, row + anchor))
