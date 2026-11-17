import io
p = "experiments/E368_s2-cross-attention-to-the-past/experiment.md"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("`cross_geometry` (8 features: cos / sin of the heading difference, the offset in the frame of the query heading along / across, the distance, e of both)", "`cross_geometry` (7 features: cos / sin of the heading difference, the offset in the frame of the query heading along / across, the distance, e of both; v1 / v2 said 8 and listed these 7)")
rep("bias MLP(8 -> 32 -> H) of the geometry", "bias MLP(7 -> 32 -> H) of the geometry")
rep("about 18 k parameters at d = 64 (a d = 96 head gives about twice that)", "17,416 parameters at d = 64 (a d = 96 head gives about twice that)")
rep("To agree (done 01:55): the hook in `jtoken.py` (Fable): `JTokenHead(..., past_block=None)` after layer 1 with past = dict(x, pos, dirv, valid, w, e), `forward(b, past=None)` returning `out['x']`, `out['w']`; defaults bit for bit unchanged.",
    "Done (hook of 81f9374b, Fable): `JTokenHead(..., past_block=None)`, the block called after layer 1 as `past_block(x, pos, dirv, valid, past)`, `forward(b, past=None)` returning `out['x']`, `out['w']`; defaults bit for bit unchanged. **Built 2026-11-17 04:00 (committed with this note):** the hook passes no `e`, so the existence probability of the CURRENT tokens travels in `past['e_cur']` (filled by `jpast.make_past`); a cold frame of arm r passes `{}` (the block runs and sees the none key only); the head's step counter advances once per OPTIMISER step (not per forward: the key-weight switch would come at step 334 otherwise); `rel_R`, `rel_t`, `has_prev` are (B, n, ...), entry k = the pose of the previous keyframe of frame k; `--seed` = the cache seed, `--init-seed` = the head initialisation and the sequence order (= `--head-seed` of `jtoken_train.py`). **A resumed run is not bit-reproducible and not batch-matched with its twin:** `trainer.fit` re-seeds and draws a fresh sampler permutation after a resume (resume steps are reported; an effect of noise size, named, not removed).")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
