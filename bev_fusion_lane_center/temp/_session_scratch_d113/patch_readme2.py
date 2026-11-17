import io
p = "training/README.md"
lines = io.open(p, encoding="utf-8").read().split("\n")
out = []
for ln in lines:
    if ln.startswith("| `data/lane_tokcache.py` |"):
        out.append("| `data/lane_tokcache.py` | D113 / E365: the predicted-node cache: the tokens of a FROZEN lane head B (a `QGraph` run: B' = `runs/e292_a_s<S>`, plain B = `runs/e273_p_s<S>`) on every TRAIN and VAL frame (frames of `TokData(split, res=2.0)`): "
                   "the K = 512 best cells by existence logit (stable order) with position, offset, heading, existence / flag / section / long-link logits and the GPU-made probabilities, the 64-d node token and the 144-d 2 m pyramid feature (float16), "
                   "and the GT labels of every token by the rule of section 2a of the E365 card (greedy assignment by descending existence within 1.5 m and heading cosine 0.5, type, raw flags, the GT links contracted over the uncovered nodes as sparse "
                   "target pairs with the gate flag, the coverage at K = 256 .. 512); one directory per run, seed and split in `runs/e365_cache/` (derived data, not in git; `manifest.json` per split); `build` (GPU, resumable, through `run.py`), "
                   "`fixture` (the cache against B's stored dump, bit for bit), `report`, `selftest` | tests `grid/test_lane_tokcache.py` |")
        out.append("| `data/lane_ptok_data.py` | E365: the loader of that cache: `loader(split, seed, bs, workers, token_drop, ident, K)` (batch: tok, pos, dirv, e, e_raw, flags, s, uout, uin, valid, ytype, dense y / ylong / yscale from the sparse pairs, gt_out, tokens_np for the "
                   "read-out writer, label = the GT side for controls only), `to_pred` (the qformat prediction of a frame from the tokens + the head's keep mask and links, no GT field), the seeded token drop of the degraded read-out | tests `grid/test_lane_tokcache.py` |")
    else:
        out.append(ln)
io.open(p, "w", encoding="utf-8").write("\n".join(out))
print("ok")
