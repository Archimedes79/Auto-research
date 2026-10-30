"""one-off: save the trained model and add the multi-keyframe recursive evaluation (--sequences) to
tests/test_flow_explains.py"""
p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1 the main block: save the model; --sequences loads a saved model and runs the sequence evaluation only
rep('''    torch.manual_seed(SEED)
    model = Model().to(DEV)
    train(model, data, iters, random.Random(SEED + 7))
    model.eval()
    r = evaluate(model, test)''', '''    torch.manual_seed(SEED)
    model = Model().to(DEV)
    saved = f"temp/checks/model_explains{TAG}.pt"                    # scratch (not in git): the trained test model
    if "--sequences" in sys.argv:                                     # evaluation only, on the model a run saved
        model.load_state_dict(torch.load(saved, map_location=DEV))
        model.eval()
        seqs = nuscenes_sequences(True, arg("--n-seq", 24), arg("--length", 5))
        sequence_report(model, seqs, f"temp/checks/flow_sequences{TAG}.png")
        sys.exit(0)
    train(model, data, iters, random.Random(SEED + 7))
    torch.save(model.state_dict(), saved)
    model.eval()
    r = evaluate(model, test)''')

# 2 sequences of consecutive keyframes from the GT cache, and the per-keyframe recursive evaluation
rep('''@torch.no_grad()
def evaluate_recursive(model, test):''', '''def _seq_meta(path):
    import numpy as np
    g = np.load(path, allow_pickle=True)["gt"].item()
    return path, g["cur"]["meta"]["scene"], g["cur"]["meta"]["timestamp"], g["cur"]["meta"]["sample"], g["prev"]["meta"]["sample"]


def nuscenes_sequences(test, n_seq, length):
    """n_seq sequences of `length` consecutive keyframes of one scene (held-out scenes if test): consecutive cached
    samples chain (the k of sample i is the k - 1 of sample i + 1). -> list of dicts: maps (T, 3, X, Y), velocity
    (T, 2, X, Y) (GT of each keyframe, its axes), steps[t] = ego steps keyframe t -> t + 1. Cached in temp/checks."""
    import glob
    import numpy as np
    path = f"temp/checks/flow_sequences_nusc_{'test' if test else 'train'}_{n_seq}_{length}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    import multiprocessing as mp
    files = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))
    with mp.get_context("fork").Pool(4) as pool:
        meta = pool.map(_seq_meta, files, chunksize=64)
    by_scene = {}
    for f, scene, ts, tok, prev_tok in meta:
        if (int(scene[:8], 16) % 5 == 0) == test:
            by_scene.setdefault(scene, []).append((ts, f, tok, prev_tok))
    out = []
    for scene in sorted(by_scene):
        rows = sorted(by_scene[scene])
        run = [rows[0]]
        for r in rows[1:]:                                   # keep only chains of directly consecutive keyframes
            run = run + [r] if r[3] == run[-1][2] else [r]
            if len(run) == length - 1:                       # length - 1 samples give length keyframes (prev of the first)
                out.append([x[1] for x in run])
                run = []
                break
        if len(out) == n_seq:
            break
    seqs = []
    for files_ in out:
        gs = [np.load(f, allow_pickle=True)["gt"].item() for f in files_]

        def maps_of(fr):
            return np.stack([np.maximum(fr["lane_divider"], fr["road_divider"]).astype(np.float32),
                             np.maximum(fr["vehicle"], fr["two_wheeler"]).astype(np.float32),
                             fr["pedestrian"].astype(np.float32)])
        frames = [gs[0]["prev"]] + [g["cur"] for g in gs]
        steps = [[(-float(dt), float(v), float(w)) for dt, v, w in reversed(g["prev"]["meta"]["steps"])] for g in gs]
        seqs.append({"maps": torch.from_numpy(np.stack([maps_of(fr) for fr in frames])),
                     "velocity": torch.from_numpy(np.stack([fr["velocity"].astype(np.float32) for fr in frames])),
                     "steps": steps})
    torch.save(seqs, path)
    return seqs


@torch.no_grad()
def sequence_report(model, seqs, image):
    """Per keyframe t of the sequences: the velocity error at F_t (vehicles, all / moving > 1 m/s) of the fresh estimate
    of the pair (F_t, F_t+1) and of the recursive one (the estimate of the previous pair carried to F_t as the START of
    this pair's refinement). Recursion pays if its error falls with t while the fresh one stays flat."""
    T = len(seqs[0]["steps"])
    acc = {(n, k, t): [0.0, 0] for n in ("fresh", "recursive") for k in ("vehicles", "moving") for t in range(T)}
    for sq in seqs:
        m, vel, steps = sq["maps"][None].to(DEV), sq["velocity"], sq["steps"]
        carried = None
        for t in range(T):
            vg = vel[t][None].to(DEV)
            fresh = model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg)[0][-1]
            recur = fresh if carried is None else model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg, v_init=carried)[0][-1]
            carried = carry(recur, [steps[t]])
            veh = sq["maps"][t, 1] > 0.5
            for k, mk in (("vehicles", veh), ("moving", veh & (vel[t].norm(dim=0) > 1.0))):
                for n, v in (("fresh", fresh), ("recursive", recur)):
                    e = (v[0].cpu() - vel[t]).norm(dim=0)
                    acc[(n, k, t)][0] += e[mk].sum().item()
                    acc[(n, k, t)][1] += int(mk.sum())
    res = {key: s_ / max(1, n_) for key, (s_, n_) in acc.items()}
    print(f"== recursive speed refinement over {len(seqs)} held-out sequences of {T + 1} keyframes (EPE m/s at F_t)")
    print("  t   vehicles fresh / recursive    moving vehicles fresh / recursive")
    for t in range(T):
        print(f"  {t}   {res[('fresh', 'vehicles', t)]:6.2f} / {res[('recursive', 'vehicles', t)]:6.2f}"
              f"            {res[('fresh', 'moving', t)]:6.2f} / {res[('recursive', 'moving', t)]:6.2f}", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for i, k in enumerate(("vehicles", "moving")):
        for n, sty in (("fresh", "o--"), ("recursive", "s-")):
            ax[i].plot(range(T), [res[(n, k, t)] for t in range(T)], sty, label=n)
        ax[i].set_xlabel("keyframe t (0.5 s apart)")
        ax[i].set_ylabel("velocity error (m/s)")
        ax[i].set_title({"vehicles": "all vehicles", "moving": "moving vehicles (> 1 m/s)"}[k], fontsize=9)
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(image, dpi=120)
    print(f"image {image}")
    return res


@torch.no_grad()
def evaluate_recursive(model, test):''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
