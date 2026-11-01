p = "autoresearch/tools/package.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)

# 1. the evaluation block and the experiment card in build()
sub('''        files = derived_files(run_dir)
        pk["deep"] = {"files"''', '''        pk["evaluation"] = evaluation_block(os.path.join(run_dir, "eval", "card.json"))
        files = derived_files(run_dir)
        pk["deep"] = {"files"''')
sub('''    return pk


def write(run_id, run_dir=None):''', '''    pk["experiment"] = experiment_of(run_id, name)
    return pk


def evaluation_block(card_path):
    """which evaluator and which frames made the numbers: from the provenance block of the evaluation card (eval/card.json); unknown when there is none"""
    out = {"evaluator": {"commit": V.UNKNOWN, "code_hash": V.UNKNOWN, "dirty": V.UNKNOWN}, "split": V.UNKNOWN, "frames": None, "frame_hash": V.UNKNOWN, "manifest_hash": V.UNKNOWN,
           "gt": {"version": V.UNKNOWN, "signature": V.UNKNOWN}, "card": V.UNKNOWN}
    try:
        prov = json.load(open(card_path, encoding="utf-8")).get("provenance") or {}
    except (OSError, ValueError, AttributeError):
        return out
    ev, data, gt = (prov.get(k) or {} for k in ("evaluator", "data", "gt"))
    dirty = ev.get("dirty")
    out.update(card=os.path.basename(os.path.dirname(os.path.dirname(card_path))) + "/eval/card.json", split=data.get("split", V.UNKNOWN), frames=data.get("n"),
               frame_hash=data.get("frame_hash", V.UNKNOWN), manifest_hash=data.get("manifest_hash", V.UNKNOWN))
    out["evaluator"] = {"commit": ev.get("commit", V.UNKNOWN), "code_hash": ev.get("code_hash", V.UNKNOWN),
                        "dirty": dirty if dirty in ([], V.UNKNOWN, None) else f"{len(dirty)} file(s) differed from the commit"}
    out["gt"] = {"version": gt.get("version", V.UNKNOWN), "signature": gt.get("signature", V.UNKNOWN)}
    return out


def experiment_of(run_id, folder):
    """the experiment card that names this run: the first experiments/*/experiment.md that holds the run id or the run folder name; unknown when none does"""
    base = os.path.join(P.root(), "experiments")
    if not os.path.isdir(base):
        return V.UNKNOWN
    for d in sorted(os.listdir(base)):
        md = os.path.join(base, d, "experiment.md")
        try:
            if os.path.isfile(md) and os.path.getsize(md) < 400_000:
                text = open(md, encoding="utf-8", errors="replace").read()
                if run_id in text or (folder and re.search(r"(?<![\w.])" + re.escape(folder) + r"(?![\w])", text)):
                    return f"experiments/{d}/experiment.md"
        except OSError:
            continue
    return V.UNKNOWN


def write(run_id, run_dir=None):''')
# 2. the weak assertion about the minimum age
sub('''        assert "younger than 0.0416" in clean(only=["x.e_s0"], min_age_days=0)[0][1] or clean(only=["x.e_s0"], min_age_days=0)[0][1] == "delete", "a minimum age of one hour is enforced"''',
    '''        mk("x.m_s0", old=0.005)                                    # about seven minutes old
        assert "younger than" in clean(only=["x.m_s0"], min_age_days=0)[0][1], "a minimum age of one hour is enforced"
        assert clean(only=["x.e_s0"], min_age_days=0)[0][1] == "delete", "12 hours are old enough for the minimum"''')
# 3. a test of the new blocks, before the best-three tests
sub('''        # no keep3.json, an empty one''', '''        json.dump({"provenance": {"evaluator": {"commit": "c1", "code_hash": "h1", "dirty": ["a.py"]}, "data": {"split": "val", "n": 7, "frame_hash": "fh", "manifest_hash": "mh"},
                                  "gt": {"version": "v5", "signature": "sg"}}}, open(os.path.join(a, "eval", "card.json"), "w"))
        os.makedirs(os.path.join(proj, "experiments", "E1_demo"))
        open(os.path.join(proj, "experiments", "E1_demo", "experiment.md"), "w").write("# E1\n\narm a: run x.a_s0 and the folder x_a_s0b\n")
        os.makedirs(os.path.join(proj, "experiments", "E2_other"))
        open(os.path.join(proj, "experiments", "E2_other", "experiment.md"), "w").write("# E2\n\nx.a_s01 is another run\n")
        pe = write("x.a_s0")
        assert pe["evaluation"] == {"evaluator": {"commit": "c1", "code_hash": "h1", "dirty": "1 file(s) differed from the commit"}, "split": "val", "frames": 7, "frame_hash": "fh",
                                    "manifest_hash": "mh", "gt": {"version": "v5", "signature": "sg"}, "card": "x_a_s0/eval/card.json"}, pe["evaluation"]
        assert pe["experiment"] == "experiments/E1_demo/experiment.md", pe["experiment"]
        assert experiment_of("x.zzz", "x_zzz") == V.UNKNOWN and evaluation_block(os.path.join(tmp, "none.json"))["split"] == V.UNKNOWN
        os.remove(os.path.join(a, "eval", "card.json"))
        # no keep3.json, an empty one''')
open(p, "w", encoding="utf-8", newline="\n".replace("\\\\", "\\")).write(s)
print("ok")
