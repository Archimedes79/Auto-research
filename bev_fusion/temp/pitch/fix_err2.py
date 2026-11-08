p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()
old = s[s.index("def load_frames(run, n, tag=\"\"):"):s.index("def load_boxes(")]
new = '''def find_store(run, n, tag=""):
    """the suite store of the evaluation: eval<tag>_n<N>, with the optional _curves / _fusion / _vel parts the flags of `eval` add to its name; refuses none or several"""
    import re
    base = os.path.join(run, "eval", "suite", "quant")
    pat = re.compile(rf"^eval{re.escape(tag)}(_curves|_fusion|_vel)*_n{n}$")
    found = sorted(d for d in os.listdir(base) if pat.match(d) and os.path.exists(os.path.join(base, d, "frames.jsonl"))) if os.path.isdir(base) else []
    if len(found) != 1:
        raise SystemExit(f"{'no' if not found else 'several'} suite store for --n {n} --tag '{tag}' in {base}: {found or sorted(os.listdir(base)) if os.path.isdir(base) else 'no quant folder'} "
                         f"(evaluate first: bev_suite.py {run} eval --n {n})")
    return os.path.join(base, found[0])


def load_frames(run, n, tag=""):
    """[(scene, index, token, label)] in the order the suite evaluated them, from the suite store of the evaluation"""
    with open(os.path.join(find_store(run, n, tag), "frames.jsonl")) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return [(r["scene"], r["index"], r["key"], r["label"]) for r in rows]


'''
s = s.replace(old, new)
# selftest: a decorated store name is found, two stores are refused
s = s.replace('''        os.makedirs(os.path.join(run, "eval", "suite", "quant", "eval_n4"))
        try:
            error_rows(run, 4)''', '''        os.rename(os.path.join(run, "eval", "suite", "quant", "eval_n3"), os.path.join(run, "eval", "suite", "quant", "eval_curves_fusion_n3"))
        assert len(load_frames(run, 3)) == 3, "the store name may carry the --curves / --fusion parts"
        try:
            find_store(run, 3, "_warm")
        except SystemExit:
            pass
        else:
            raise AssertionError("another tag must not find the store")
        os.makedirs(os.path.join(run, "eval", "suite", "quant", "eval_n4"))
        try:
            error_rows(run, 4)''')
open(p, "w", encoding="utf-8").write(s)
