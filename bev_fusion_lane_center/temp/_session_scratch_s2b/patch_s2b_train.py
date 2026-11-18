"""E387 (S2b) edits of the restored trainer and its test in a scratch tree: --block {residual,fused}. usage: python patch_s2b_train.py <tree root>"""
import os
import sys

root = os.path.abspath(sys.argv[1])


def edit(rel, pairs):
    p = os.path.join(root, rel)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)


edit("training/train/jpast_train.py", [
    ('RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past", "block")\nBLOCKS = {"residual": JP.PastBlock, "fused": JP.FusedPastBlock}           # E387: --block; the configs of E368 have no such key and are the residual block'),
    ('    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic sequences")', '    ap.add_argument("--block", choices=sorted(BLOCKS), default="residual", help="arm r/f: residual = E368\'s PastBlock (the default), fused = E387\'s FusedPastBlock (cross-attention + gated MLP fusion)")\n    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic sequences")'),
    ('    assert args["bs"] > 0 and args["steps"] > 0 and args["lr"] > 0,', '    assert not (args["no_past"] and (args.get("block") or "residual") == "fused"), "--no-past (arm z) has no block: --block fused contradicts it"\n    assert args["bs"] > 0 and args["steps"] > 0 and args["lr"] > 0,'),
    ('        old = {k: (c.get("args") or {}).get(k) for k in RECIPE}\n        assert c.get("design") == "jpast" and old == {k: args.get(k) for k in RECIPE},', '        old = recipe_of(c.get("args") or {})\n        assert c.get("design") == "jpast" and old == recipe_of(args),'),
    ('def check_start(args, folder):', 'def recipe_of(d):\n    """the recipe keys of an args dict; a missing block is the residual block (the configs of E368)"""\n    r = {k: d.get(k) for k in RECIPE}\n    r["block"] = r["block"] or "residual"\n    return r\n\n\ndef check_start(args, folder):'),
    ('        head.past_block = JP.PastBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1])', '        head.past_block = BLOCKS[args.get("block") or "residual"](d=JKT.arch({})[0], heads=JKT.arch({})[1])'),
    ("'arm r: the head + PastBlock, '", "'arm r: the head + ' + type(model.past_block).__name__ + ', '"),
])

edit("training/grid/test_jpast_train.py", [
    ('''    (T0, "the block is built before the head", "    head = JKT.build_head({})\\n    if not args.get(\\"no_past\\"):\\n        check_hook(head)\\n        head.past_block = JP.PastBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1])\\n    return head",
     "    blk = JP.PastBlock(d=JKT.arch({})[0], heads=JKT.arch({})[1]) if not args.get(\\"no_past\\") else None''',
     '''    (T0, "the block is built before the head", "    head = JKT.build_head({})\\n    if not args.get(\\"no_past\\"):\\n        check_hook(head)\\n        head.past_block = BLOCKS[args.get(\\"block\\") or \\"residual\\"](d=JKT.arch({})[0], heads=JKT.arch({})[1])\\n    return head",
     "    blk = BLOCKS[args.get(\\"block\\") or \\"residual\\"](d=JKT.arch({})[0], heads=JKT.arch({})[1]) if not args.get(\\"no_past\\") else None'''),
    ('''(T0, "a change of seq is not refused at a resume", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_past")'),''',
     '''(T0, "a change of seq is not refused at a resume", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past", "block")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "steps", "lr", "no_past", "block")'),
    (T0, "the block is not part of the recipe", 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past", "block")', 'RECIPE = ("seed", "init_seed", "cache", "ident", "K", "bs", "seq", "steps", "lr", "no_past")'),
    (T0, "the fused option builds the residual block", 'BLOCKS = {"residual": JP.PastBlock, "fused": JP.FusedPastBlock}', 'BLOCKS = {"residual": JP.PastBlock, "fused": JP.PastBlock}'),
    (T0, "a config without a block is not the residual block", 'r["block"] = r["block"] or "residual"', 'r["block"] = r["block"]'),
    (T0, "arm z accepts a block", 'assert not (args["no_past"] and (args.get("block") or "residual") == "fused"), ', 'assert True or '),'''),
    ("    sz, sr = hz.state_dict(), hr.state_dict()\n", '''    torch.manual_seed(5)
    hf = T.build_head(dict(no_past=False, block="fused"))
    sz, sr = hz.state_dict(), hr.state_dict()
    assert isinstance(hf.past_block, JP.FusedPastBlock) and isinstance(T.build_head(dict(no_past=False, block=None)).past_block, JP.PastBlock) and isinstance(hr.past_block, JP.PastBlock), f"{label}: --block picks the block"
    assert all(torch.equal(sz[k], hf.state_dict()[k]) for k in sz), f"{label}: the fused arm does not start from the same head"
    refuses(lambda: T.check_options(dict(bs=4, steps=1, lr=1e-3, seq=3, no_past=True, block="fused")), "contradicts")
'''),
    ('''        change = dict(seed=lambda v: v + 1, init_seed=lambda v: v + 1, cache=lambda v: "another_root", ident=lambda v: "e273_p", K=lambda v: 384, bs=lambda v: v + 1, seq=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2, no_past=lambda v: not v)''',
     '''        change = dict(seed=lambda v: v + 1, init_seed=lambda v: v + 1, cache=lambda v: "another_root", ident=lambda v: "e273_p", K=lambda v: 384, bs=lambda v: v + 1, seq=lambda v: v + 1, steps=lambda v: v + 1, lr=lambda v: v * 2, no_past=lambda v: not v,
                      block=lambda v: "fused" if v != "fused" else "residual")'''),
    ('''        T.check_start(cfg, folder)                                                                       # the same recipe is accepted''',
     '''        T.check_start(cfg, folder)                                                                       # the same recipe is accepted
        old_cfg = json.load(open(os.path.join(folder, "config.json")))                                   # E368's configs have no block key = the residual block
        del old_cfg["args"]["block"]
        os.makedirs(os.path.join(tmp, "old"))
        json.dump(old_cfg, open(os.path.join(tmp, "old", "config.json"), "w"))
        T.check_start(dict(cfg, block="residual"), os.path.join(tmp, "old"))
        refuses(lambda: T.check_start(dict(cfg, block="fused"), os.path.join(tmp, "old")), "of another run")
        af = T.parse_args(["--name", "f", "--smoke", "--bs", "2", "--seq", "3", "--seed", "0", "--block", "fused"])      # the fused arm trains 20 steps on the smoke batches
        T.train(af, runs=tmp, ckpt_every=10, smoke_k=K)
        mf, argsf = T.load_model("f", "cpu", tmp)
        assert isinstance(mf.past_block, JP.FusedPastBlock) and argsf["block"] == "fused" and mf._step_py == 20, f"{label}: the fused smoke run"'''),
])
print("patched", root)
