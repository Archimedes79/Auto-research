import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def cut(start_marker, end_marker, new):
    global s
    a = s.index(start_marker)
    b = s.index(end_marker)
    s = s[:a] + new + s[b:]


cut("def check_parity(", "def head_macs(", '''def compare_sets(m, cb, cells_c, live, tol, limit=UNMATCHED_LIMIT):
    """the head on the cache batch `cb` (cells_c (B, K): its cells) and on the live batch `live` (inputs + `cell`), the live tokens matched to the cache's by cell (`align`) -> (ok, numbers). PASS when p_none and every link probability of the
    valid tokens agree within `tol` AND the valid tokens that only one side has (a cache token without a valid live match, a live token the cache lacks) are at most `limit` x the cache's valid tokens"""
    perm = align(cells_c, live["cell"].long().numpy())
    live_c = reorder(live, perm)                                                   # the live tokens in the cache's order
    ok, num = compare_outputs(run_head(m, cb, "cpu"), run_head(m, live_c, "cpu"), tol)
    v = cb["valid"] & live_c["valid"]
    fields = {k: float((cb[k].float() - live_c[k].float()).abs()[v].max()) for k in JK.INPUTS if k != "valid"}
    n_c, n_l = int(cb["valid"].sum()), int(live["valid"].sum())
    only = int((cb["valid"] & ~live_c["valid"]).sum()) + n_l - int(live_c["valid"].sum())
    num.update(valid_cache=n_c, valid_live=n_l, valid_cache_not_live=int((cb["valid"] & ~live_c["valid"]).sum()), valid_live_not_cache=n_l - int(live_c["valid"].sum()), unmatched_share=only / max(n_c, 1), unmatched_limit=limit,
               input_max_abs_diff_matched_valid_tokens=fields)
    return bool(ok and only <= limit * n_c), num


def check_parity(a, runs=RUNS):
    """the head on the first `--n` VAL frames, once from the cache tokens (fp16 `tok`) and once from tokens recomputed live from B' on the CPU in float32 (`live_tokens`, compared by cell, `compare_sets`): PASS when p_none and every link
    probability of the valid tokens agree within --tol and the valid tokens that only one side has are at most UNMATCHED_LIMIT (0.5 %) of the cache's valid tokens; the input fields of the matched valid tokens are reported. Writes
    runs/R/token_parity_<UTC>.json (a new file every time) -> 0 / 1"""
    import lane_tok_data as TD
    import qpredict as QP
    m, args = JKT.load_model(a.run, "cpu", runs)
    kw = cache_kw(args)
    ds = open_cache("val", kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    n = a.n
    b_run = _abs(a.b_run or f"runs/{kw['ident']}_s{kw['seed']}")
    mb = QP.load_model(b_run, "cpu")
    td = TD.TokData("val", LATTICE, feats=True)
    assert list(td.tokens[:n]) == list(ds.tokens[:n])
    cb = next(iter(batches(ds, range(n), bs=n)))
    x = torch.stack([td[i]["x"] for i in range(n)])
    with torch.no_grad():
        live = live_tokens(mb, x, ds.K, exact=True)
    ok, num = compare_sets(m, cb, np.stack([t["cell"] for t in cb["tokens_np"]]), live, a.tol)
    num.update(frames=n, run=a.run, b_run=os.path.relpath(b_run, ROOT), ok=ok, note="cache: tok fp16, built on the GPU; live: CPU, tok fp32; tokens are matched by cell, padding tokens are not compared")
    print(json.dumps(num, indent=1))
    print(f"{'PASS' if ok else 'FAIL'}: parity of the head on {n} VAL frames, cache against live B' ({a.run}), tolerance {a.tol}, unmatched limit {UNMATCHED_LIMIT}; written {os.path.relpath(_write_stamped(os.path.join(runs, a.run), 'token_parity', num), ROOT)}")
    return int(not ok)


''')

cut("def gpu_free(", "# ------------------------------------------------------------------------------------------------------------------------------ command line", '''def _ancestors():
    """the pids of this process and its parents (a timing started as a run.py job has run.py among them)"""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def gpu_free(limit_mib=2000):
    """(ok, why): the timing rule of the card: nvidia-smi shows less than `limit_mib` used and no run.py job is active, not counting this process and its ancestors (the run.py job this timing may be)"""
    try:
        used = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().split("\\n")[0])
        jobs = [j for j in subprocess.run(["pgrep", "-f", "autoresearch/tools/run.py"], capture_output=True, text=True, timeout=20).stdout.split() if int(j) not in _ancestors()]
    except Exception as e:  # noqa: BLE001
        return False, f"cannot read the GPU: {e}"
    if used >= limit_mib or jobs:
        return False, f"GPU memory used {used:.0f} MiB (limit {limit_mib}), run.py jobs {jobs}"
    return True, f"GPU memory used {used:.0f} MiB, no other run.py job"


def timing(a, runs=RUNS):
    """parameters, GMACs and ms per frame of the head (batch 1, bf16 autocast on the GPU, CUDA events, `--frames` cached VAL frames, `--repeats` passes after one untimed pass) and, with --b-run, of B' alone (pyramid + node head + top-K gather) and of
    the whole path (B' + head in one timed region). K = --K or the K of the cache the run was trained on (at most that). The GPU only when `gpu_free`; --cpu times the code path on the CPU. Writes runs/R/token_timing[_cpu]_<UTC>.json (a new
    file every time) with `quiet` (host load < 3 and an idle GPU before and after) and `usable_for_gcost` (true only for a quiet GPU result)"""
    import jrecur_time as JRT
    import jtiming as JTG
    m, args = JKT.load_model(a.run, "cpu", runs)
    ok, why = (False, "--cpu") if a.cpu else gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why}); --cpu times the code path on the CPU only")
    dev = "cpu" if a.cpu else "cuda"
    kw = cache_kw(args)
    ds = open_cache("val", kw["seed"], kw["ident"], kw["K"], kw.get("cache"))
    K = a.K or ds.K
    assert K <= ds.K, f"--K {K} is above the K of the cache ({ds.K})"
    gmacs = head_macs(m, K)                                                            # on the CPU, before the head moves
    m = m.to(dev)
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    cut = lambda b: {k: v[:, :K] for k, v in b.items() if k in JK.INPUTS}  # noqa: E731
    items = [JKT.to_device(cut(LP.collate([ds[i]])), dev) for i in range(a.frames)]
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")  # noqa: E731
    mb = xs = None
    if a.b_run:
        import lane_tok_data as TD
        import qpredict as QP
        mb = QP.load_model(_abs(a.b_run), dev)
        td = TD.TokData("val", LATTICE, feats=True)
        xs = [td[i]["x"][None].to(dev) for i in range(a.frames)]
    ms = dict(head=[], bprime=[], whole=[])
    for rep in range(a.repeats + 1):
        row = dict(head=[], bprime=[], whole=[])
        for i, b in enumerate(items):
            with torch.no_grad(), ac():
                row["head"].append(JRT.timed(lambda: m(b), dev == "cuda")[1])
                if mb is not None:
                    row["bprime"].append(JRT.timed(lambda: live_tokens(mb, xs[i], K), dev == "cuda")[1])
                    row["whole"].append(JRT.timed(lambda: m(live_tokens(mb, xs[i], K)), dev == "cuda")[1])
        if rep:                                                                           # the first pass is untimed
            for k in ms:
                ms[k].append(row[k])
    load1, gpu1 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    res = dict(run=a.run, device=dev, K=K, frames=a.frames, repeats=a.repeats, params=JK.count_params(m), gmacs_head=gmacs, host=line, quiet=bool(quiet), usable_for_gcost=bool(quiet and dev == "cuda"), gpu_rule=why, bf16=dev == "cuda",
               median_ms={k: float(np.median(v)) for k, v in ms.items() if v}, p90_ms={k: float(np.percentile(v, 90)) for k, v in ms.items() if v}, macs_note=MACS_NOTE)
    if mb is not None:
        res["params_bprime"] = JK.count_params(mb.pyr) + JK.count_params(mb.nodes)           # the path of the tokens: pyramid + node head (the link, map and assign heads of the checkpoint are not on it)
        res["params_whole_path"] = res["params_bprime"] + res["params"]
    print(json.dumps(res, indent=1))
    print(f"written {os.path.relpath(_write_stamped(os.path.join(runs, a.run), 'token_timing' + ('_cpu' if a.cpu else ''), res), ROOT)}")
    return 0


''')
old = 's.add_argument("--K", type=int, default=384)'
assert s.count(old) == 1
s = s.replace(old, 's.add_argument("--K", type=int, default=None, help="tokens of the head (default: the K of the cache of the run, 384 in the card)")')
open(p, "w", encoding="utf-8").write(s)
