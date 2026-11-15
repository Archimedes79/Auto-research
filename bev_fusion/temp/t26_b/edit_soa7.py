p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()

# ---- (5) gt identity: unknown = refused
a = s.index("def gt_identity(run, signature=None, allow_mismatch=False):")
b = s.index("def select_frames(")
new = '''def trained_gt(run):
    """the GT signature a run was trained on (RUN/gt.json), None when that cannot be told: no run, no folder, no file, no key"""
    try:
        with open(os.path.join(run, "gt.json")) as f:
            sig = json.load(f).get("gt_signature")
    except (TypeError, OSError, ValueError, AttributeError):
        return None
    return sig if isinstance(sig, str) and sig else None


def check_gt(sig, trained, allow_mismatch=False):
    """True = the GT scored against (signature `sig`) is the GT the run was trained on (`trained`: its cache key or a signature; a version that makes the same GT counts), False = another one,
    None = unknown. Another GT AND an unknown one are REFUSED (SystemExit) unless allow_mismatch: an unchecked identity is no identity"""
    from GtGenerator import versions
    match = None if trained is None else versions.canonical(trained.rsplit("_", 1)[-1]) == versions.canonical(sig)
    if match is not True and not allow_mismatch:
        what = "the GT the run was trained on is unknown (no runs/RUN/gt.json with a gt_signature)" if match is None else f"the run was trained on {trained}"
        raise SystemExit(f"refused: {what}, the GT scored against is {sig}; pass --allow-gt-mismatch to score the run on it anyway (the json records it)")
    return match


def gt_identity(run, signature=None, allow_mismatch=False):
    """the GT that is scored against: the stored GT of `signature` (default the live generator's), printed with its version; refused (before anything is read) when it is not the GT the run was
    trained on (runs/RUN/gt.json) or when that is unknown, unless allow_mismatch; -> {signature, version, store, run_signature, match, note}"""
    from GtGenerator import current_signature, versions
    sig = signature or current_signature()
    trained = trained_gt(run)
    match = check_gt(sig, trained, allow_mismatch)
    store = os.path.join(WS, "data", "autogt", versions.canonical(sig))
    if not os.path.isdir(store):
        raise SystemExit(f"no stored GT {store} for the signature {sig}")
    info = {"signature": sig, "version": versions.version_of_key(sig), "store": os.path.relpath(store, WS), "run_signature": trained, "match": match,
            "note": "the live GT" if signature is None else f"the GT of signature {sig}"}
    print(f"GT scored against: {info['version']} signature {sig} (store {info['store']}); the run was trained on {trained} -> " + {True: "the same GT", None: "UNKNOWN", False: "A DIFFERENT GT"}[match], flush=True)
    if match is not True:
        info["note"] += f"; the run was trained on {trained or 'an unknown GT'} (--allow-gt-mismatch)"
    return info


'''
s = s[:a] + new + s[b:]

# ---- (9) missing frames reported
old = s[s.index("def select_frames("):s.index("@contextlib.contextmanager")]
new_sel = '''def select_frames(universe, n=None, only=None):
    """universe = [{i (row of the logits), token, scene}]; only = the tokens of a stored result (their order); n = that many frames spread evenly over what is left.
    -> (the frames, the tokens of `only` that are not in the universe: they cannot be scored and are reported, never dropped silently)"""
    have = {u["token"] for u in universe}
    missing = [t for t in (only or []) if t not in have]
    sel = [u for u in universe if only is None or u["token"] in only]
    if only is not None:
        order = {t: k for k, t in enumerate(only)}
        sel.sort(key=lambda u: order[u["token"]])
    if n and n < len(sel):
        sel = sel[:: max(1, len(sel) // n)][:n]
    return sel, missing


'''
s = s.replace(old, new_sel)
s = s.replace('''    sel = select_frames(universe, a.n, only)
    if not sel:''', '''    sel, missing = select_frames(universe, a.n, only)
    if missing:
        print(f"WARNING: {len(missing)} of the {len(only)} frames of {a.frames_from} are not samples of the run's val set (or have no cached reference logits) and are NOT scored", flush=True)
    if not sel:''')
s = s.replace('''"offset_m": a.offset, "claim_rows": rows,''', '''"offset_m": a.offset, "claim_rows": rows, "frames_requested": len(only) if only else None, "frames_missing": len(missing), "frames_missing_tokens": missing[:10],''')

# ---- (9) in-sample flag on d2
s = s.replace('''                        row[d] = {"ours": r["b"], "ref": r["a"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_valid": r["n_valid"]}''',
              '''                        row[d] = {"ours": r["b"], "ref": r["a"], "diff": r["diff"], "lo": r["lo"], "hi": r["hi"], "n_valid": r["n_valid"], **({"in_sample_thresholds": True} if d == "d2" else {})}''')
open(p, "w", encoding="utf-8").write(s)
