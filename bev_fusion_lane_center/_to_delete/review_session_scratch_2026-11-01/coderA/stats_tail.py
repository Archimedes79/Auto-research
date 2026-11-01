# ------------------------------------------------------------------------------------------------ the paired scene bootstrap
# ONE implementation for the whole repository (D24): the evalsuite's compare, the lane project's sparse_lanes/bootstrap.py,
# topo_metrics.paired and drivers/boot_eval.py all draw their scenes here and take their intervals here.

def scene_picks(n_scenes: int, n: int = 1000, seed: int = 0) -> Iterator[np.ndarray]:
    """THE random stream of every scene bootstrap: per draw `n_scenes` scene positions drawn with replacement,
    np.random.RandomState(seed).randint(0, n_scenes, n_scenes), draw after draw. The lane project's stored intervals rest on exactly
    this sequence (sparse_lanes/bootstrap.py since loop 13): changing it changes every stored interval."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        yield rng.randint(0, n_scenes, n_scenes)


def scene_draws(scenes: Sequence[Any], n: int = 1000, seed: int = 0) -> Iterator[np.ndarray]:
    """frame indices of every draw. `scenes` = the scene of every frame (one label per frame, any order). The scenes are ordered as
    np.unique orders them; each draw takes scene_picks() of them and returns the indices of their frames, concatenated in draw order
    (a scene drawn twice is in the draw twice)."""
    ids, inv = np.unique(np.asarray(scenes), return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    members = [np.nonzero(inv == s)[0] for s in range(len(ids))]
    for pick in scene_picks(len(ids), n, seed):
        yield np.concatenate([members[s] for s in pick])


def percentile_interval(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float, int]:
    """(lo, hi, n_valid): the alpha / 2 and 1 - alpha / 2 percentiles of the finite values. Non-finite draws (a resample with nothing
    to score, e.g. no ground truth beyond 40 m) are dropped and COUNTED: the interval is then conditional on the n_valid scorable draws,
    and a caller must report n_valid when it is below the number of draws."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 - 100 * alpha / 2])
    return float(lo), float(hi), int(len(v))


def paired_draws(stat_a: Callable[[Any], Dict[str, float]], stat_b: Callable[[Any], Dict[str, float]], draws: Iterable[Any],
                 keys: Sequence[str], alpha: float = 0.05, better: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """THE paired interval: the difference a - b of two systems on the same draws. Convention of the repository: a = the variant,
    b = the control, so `diff`, `lo`, `hi` are variant - control (what autoresearch/tools/decide.py reads from a card's
    compare.paired.<metric>.lo / hi). stat_x(sel) -> {key: number} of system x on the selection `sel` of one draw (sel None = all
    frames); `draws` = scene_draws(...) (frame indices) or scene_picks(...) (scene positions) or anything stat_x understands.
    -> key -> {a, b, diff (a - b on all frames), lo, hi (interval of a - b over the draws), p_gt0 (share of the valid draws with
    a - b > 0), p_a_better (share of the valid draws in which a is strictly better: a - b > 0 for a 'high' key, < 0 for a 'low' key of
    `better`, default 'high'), n_valid (finite draws), n_draws}. A key missing on all frames, or in every draw, is left out."""
    base_a, base_b = stat_a(None), stat_b(None)
    d: Dict[str, List[float]] = {k: [] for k in keys}
    for sel in draws:
        ma, mb = stat_a(sel), stat_b(sel)
        for k in keys:
            if k in ma and k in mb:
                d[k].append(ma[k] - mb[k])
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        if k not in base_a or k not in base_b or not d[k]:
            continue
        v = np.asarray(d[k], float)
        lo, hi, nv = percentile_interval(v, alpha)
        f = v[np.isfinite(v)]
        high = (better or {}).get(k, "high") == "high"
        out[k] = {"a": float(base_a[k]), "b": float(base_b[k]), "diff": float(base_a[k]) - float(base_b[k]), "lo": lo, "hi": hi,
                  "p_gt0": float((f > 0).mean()) if nv else float("nan"),
                  "p_a_better": float(((f > 0) if high else (f < 0)).mean()) if nv else float("nan"),
                  "n_valid": nv, "n_draws": int(len(v))}
    return out


def interval_draws(stat: Callable[[Any], Dict[str, float]], draws: Iterable[Any], keys: Sequence[str], alpha: float = 0.05
                   ) -> Dict[str, Dict[str, Any]]:
    """one system: key -> {value (all frames), lo, hi (interval over the draws), sd (of the finite draws), n_valid, n_draws};
    stat and draws as in paired_draws"""
    base = stat(None)
    vals: Dict[str, List[float]] = {k: [] for k in keys}
    for sel in draws:
        m = stat(sel)
        for k in keys:
            if k in m:
                vals[k].append(m[k])
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        if k not in base or not vals[k]:
            continue
        v = np.asarray(vals[k], float)
        lo, hi, nv = percentile_interval(v, alpha)
        out[k] = {"value": base[k], "lo": lo, "hi": hi, "sd": float(np.std(v[np.isfinite(v)])) if nv else float("nan"),
                  "n_valid": nv, "n_draws": int(len(v))}
    return out


def bootstrap_scenes(a: Dict[str, list], b: Dict[str, list], reduce: Callable[..., Dict[str, float]], keys: Sequence[str],
                     n: int = 1000, seed: int = 0, alpha: float = 0.05, better: Optional[Dict[str, str]] = None
                     ) -> Dict[str, Dict[str, Any]]:
    """Paired bootstrap over scenes of two stored runs: a = the CONTROL (reference), b = the VARIANT; a, b: scene -> list of Counts
    (only the scenes in both are used). Each draw samples scenes with replacement (scene_picks, the same draw for both), reduce() gives
    both runs' numbers, the difference b - a = variant - control is recorded (paired_draws(b, a): the sign decide.py and the lane
    project's SB.paired(variant, control) use). -> key -> {a, b, diff (b - a), lo, hi, p_b_better (share of the valid draws in which
    b is strictly better, by the direction in `better`, default 'high'), p_diff_gt0 (share with b - a > 0), n_valid (finite draws),
    n_draws, n_scenes}. reduce(counts, keys=keys) must accept the keys argument."""
    scenes = sorted(set(a) & set(b))
    if not scenes:
        raise ValueError("the two runs have no scene in common")

    def stat(x):
        return lambda pick: reduce([c for i in (range(len(scenes)) if pick is None else pick) for c in x[scenes[i]]], keys=keys)
    res = paired_draws(stat(b), stat(a), scene_picks(len(scenes), n, seed), keys, alpha, better)
    return {k: {"a": v["b"], "b": v["a"], "diff": v["diff"], "lo": v["lo"], "hi": v["hi"], "p_b_better": v["p_a_better"],
                "p_diff_gt0": v["p_gt0"], "n_valid": v["n_valid"], "n_draws": v["n_draws"], "n_scenes": len(scenes)} for k, v in res.items()}
