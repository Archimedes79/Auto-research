set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/grid/test_jpast_train.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''# ------------------------------------------------------------------------------------------------------------------------------ mutants
def one(args):''', '''def real_chain_check():
    """one batch of the real TRAIN chains (the cache and its extension, 2 sequences of 3 keyframes) through the loop with the real head on the CPU, and the sequence order of two arms with the same seeds (not run for the mutants)"""
    import torch
    import trainer as TR
    import lane_past_data as LPD
    T = load(TRAIN, "jpast_train_real")
    ds, dl = LPD.loader("train", 0, 2, 0)
    TR.seed_all(0)
    b = next(iter(dl))
    rows = torch.stack([f["row"] for f in b["frames"]], 1)
    assert tuple(rows.shape) == (2, 3) and (rows[:, 1:] - rows[:, :-1] == 1).all() and b["has_prev"][:, 1:].all() and tuple(b["rel_t"].shape) == (2, 3, 2) and b["frames"][0]["tok"].shape[1] == ds.tok.K, "real chains"
    torch.manual_seed(0)
    m = T.build_head(dict(no_past=False))
    for p in (m.past_block.o.weight, m.past_block.o.bias):
        torch.nn.init.normal_(p, std=0.3)
    m.train()
    m.set_step(0)
    loss, infos = T.sequence_loss(m, b, "cpu")
    loss.backward()
    assert torch.isfinite(loss) and len(infos) == 3 and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in m.past_block.parameters()), "the real chains through the loop"

    def first(seed):
        TR.seed_all(seed)
        return [bb["idx"].tolist() for _, bb in zip(range(3), LPD.loader("train", 0, 4, 0)[1])]
    assert first(0) == first(0) and first(0) != first(1), "the sequence order follows the seed"
    print(f"real chains ok: loss {float(loss):.3f}, {len(ds)} chains, first batch rows {rows.tolist()}")


# ------------------------------------------------------------------------------------------------------------------------------ mutants
def one(args):''')
rep('''    if len(sys.argv) > 1 and sys.argv[1] == "one":
        selftest(sys.argv[2], sys.argv[3])
        sys.exit(0)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", TRAIN, EVAL], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"the selftest failed:\n{r.stdout[-1500:]}{r.stderr[-2500:]}"
    print(r.stdout.strip())''', '''    if len(sys.argv) > 1 and sys.argv[1] in ("one", "real"):
        selftest(sys.argv[2], sys.argv[3]) if sys.argv[1] == "one" else real_chain_check()
        sys.exit(0)
    for args_ in (["one", TRAIN, EVAL], ["real"]):
        r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args_, capture_output=True, text=True, env=ENV)
        assert r.returncode == 0, f"{args_[0]} failed:\n{r.stdout[-1500:]}{r.stderr[-2500:]}"
        print(r.stdout.strip().splitlines()[-1])''')
rep('''    wsl bash tools/env/_py14.sh test_jpast_train.py selftests  # the selftest only''', '''    wsl bash tools/env/_py14.sh test_jpast_train.py selftests  # the selftest and one batch of the real TRAIN chains only''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
