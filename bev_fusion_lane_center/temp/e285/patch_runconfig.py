p = "training/train/run_config.py"
s = open(p, encoding="utf-8").read()
a = s[s.index("FLAGS = {"):s.index("def main():")]
b = '''FLAGS = {"grid_head": ("name", "seed", "steps", "bs", "lr", "width", "workers", "budget", "per100"),
         "qgraph_head": ("name", "seed", "phases", "steps", "bs", "workers", "width", "head", "resume_from", "gate_phases", "ev_w", "node_w_evt", "node_w_inter", "long_match", "tier_w", "vis_w",
                         "neg_w", "link_tier_w", "sparse", "node_loss", "zone_bias", "soft_nms")}
SWITCHES = ("smoke", "no_gates")                    # a key that is true becomes a flag without a value


def argv_of(cfg):
    """the argument list of the trainer of cfg["design"] (a list of strings); a key `node_w_evt` is the flag `--node-w-evt`"""
    design = cfg.get("design")
    if design not in FLAGS:
        raise SystemExit(f"design {design!r} is not one of {sorted(FLAGS)}")
    out = []
    for k in FLAGS[design]:
        if k not in cfg:
            continue
        v = cfg[k]
        out += ["--" + k.replace("_", "-")] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    for k in SWITCHES:
        if cfg.get(k):
            out.append("--" + k.replace("_", "-"))
    return out


'''
s = s.replace(a, b)
a2 = '''        assert "--smoke" not in argv_of(dict(design="grid_head", name="x", steps=3))'''
b2 = '''        assert "--smoke" not in argv_of(dict(design="grid_head", name="x", steps=3))
        q = argv_of(dict(design="qgraph_head", name="e273_p_s1", seed=1, phases=[3, 4], steps=[10, 20], node_w_evt=3.0, tier_w=[1, 2, 3], no_gates=True, workers=2))
        assert q == ["--name", "e273_p_s1", "--seed", "1", "--phases", "3", "4", "--steps", "10", "20", "--workers", "2", "--node-w-evt", "3.0", "--tier-w", "1", "2", "3", "--no-gates"], q'''
assert a2 in s
s = s.replace(a2, b2)
a3 = '''    if cfg["design"] == "grid_head":
        import grid_train as GT
        sys.argv = ["grid_train.py"] + argv_of(cfg)
        GT.main()'''
b3 = '''    mod = {"grid_head": "grid_train", "qgraph_head": "qgraph_train"}[cfg["design"]]
    trainer = __import__(mod)
    sys.argv = [mod + ".py"] + argv_of(cfg)
    trainer.main()'''
assert a3 in s
s = s.replace(a3, b3)
open(p, "w", encoding="utf-8").write(s)
