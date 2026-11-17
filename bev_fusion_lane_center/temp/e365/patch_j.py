import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep('''import argparse
import json
import os
import sys
import time

import torch
''', '''import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
''')
rep('''RECIPE = ("seed", "cache", "ident", "K", "bs", "steps", "lr")''', '''RECIPE = ("seed", "cache", "ident", "K", "bs", "steps", "lr", "d", "layers", "heads", "ffn", "token_drop")
ARCH_DEFAULTS = dict(d=64, layers=3, heads=4, ffn=None, token_drop=0.0)     # what a run of before these options means (the head of the first build); ffn None = 2 x d''')
rep('''    ap.add_argument("--smoke"''', '''    ap.add_argument("--d", type=int, default=ARCH_DEFAULTS["d"], help="width of the head")
    ap.add_argument("--layers", type=int, default=ARCH_DEFAULTS["layers"], help="self-attention layers")
    ap.add_argument("--heads", type=int, default=ARCH_DEFAULTS["heads"], help="attention heads (d must be a multiple)")
    ap.add_argument("--ffn", type=int, default=ARCH_DEFAULTS["ffn"], help="feed-forward width (default 2 x d)")
    ap.add_argument("--token-drop", type=float, default=ARCH_DEFAULTS["token_drop"], dest="token_drop", help="train-time removal of each valid token with this probability, a new draw every step (seeded by seed, step, frame token)")
    ap.add_argument("--smoke"''')
rep('''        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}''', '''        old = {k: (c.get("args") or {}).get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}                    # a run of before the architecture options has the defaults''')
rep('''def make_loss(run):
    """`jtoken.jtoken_loss` + what the log needs: every step the peak GPU memory; at call 50 and every 100 calls a clock line (seconds per step since the last line)"""''', '''def build_head(args):
    """the head of a run's recipe (its args: d, layers, heads, ffn, with the defaults of a run of before these options): the one place that builds it, for the trainer and `load_model`"""
    g = lambda k: args.get(k) if args.get(k) is not None else ARCH_DEFAULTS[k]  # noqa: E731
    d = g("d")
    return JK.JTokenHead(d=d, heads=g("heads"), layers=g("layers"), ffn=args.get("ffn") or 2 * d)


def step_keep(tokens, K, p, seed, step):
    """(B, K) bool: the tokens that stay in a step: each is removed with probability `p` by a generator seeded with (seed, step, crc32 of the frame's token): a new draw every step, the same on a resume"""
    return np.stack([np.random.default_rng([seed, step, zlib.crc32(t.encode())]).random(K) >= p for t in tokens])


def drop_tokens(b, keep):
    """the batch with the tokens outside `keep` (B, K) removed, as the loader's `token_drop` does it (`lane_ptok_data.PTokData`): not valid, type none, every target pair with a removed end gone (y, ylong, yscale); the targets are NOT re-contracted
    over the removed tokens (the GT links through a removed token are lost, not bridged)"""
    keep = keep.to(b["valid"].device)
    out = dict(b, valid=b["valid"] & keep, ytype=torch.where(keep, b["ytype"], torch.zeros_like(b["ytype"])))
    pair = keep[:, :, None] & keep[:, None, :]
    out["y"] = torch.where(pair, b["y"], torch.zeros_like(b["y"]))
    out["ylong"] = b["ylong"] & pair
    out["yscale"] = torch.where(pair[..., None], b["yscale"], torch.full_like(b["yscale"], float("nan")))
    return out


def make_loss(run, token_drop=0.0, seed=0):
    """`jtoken.jtoken_loss` + what the log needs: every step the peak GPU memory; at call 50 and every 100 calls a clock line (seconds per step since the last line); token_drop > 0: `drop_tokens` with a draw of `step_keep` per step
    (the step is the head's own training-step counter, which a resume and a roll-back keep)"""''')
rep('''        b = to_device(b, next(m.parameters()).device)
        loss, info''', '''        b = to_device(b, next(m.parameters()).device)
        if token_drop > 0:
            toks = b.get("token") or [f"frame{i}" for i in range(b["valid"].shape[0])]                    # a synthetic batch has no frame tokens
            b = drop_tokens(b, torch.from_numpy(step_keep(toks, b["valid"].shape[1], token_drop, seed, m._step_py)))
        loss, info''')
rep('''    m = JK.JTokenHead()
    m.load_state_dict(ck["state"])''', '''    m = build_head(ck["args"])
    m.load_state_dict(ck["state"])''')
rep('''    model = JK.JTokenHead()
    run.log(f"jtoken bs''', '''    model = build_head(args)
    run.log(f"jtoken bs''')
rep('''trainable {JK.count_params(model) / 1e6:.3f} M (the whole head; B' is cached), "''', '''trainable {JK.count_params(model) / 1e6:.3f} M (the whole head, d {args['d']} layers {args['layers']} heads {args['heads']} ffn {args['ffn'] or 2 * args['d']}; B' is cached), "
            + (f"token drop {a.token_drop} per step, " if a.token_drop > 0 else "")''')
rep('''T.fit(model, make_loss(run),''', '''T.fit(model, make_loss(run, a.token_drop, a.seed),''')
open(p, "w", encoding="utf-8").write(s)
