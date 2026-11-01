"""The best three networks of a problem: which ones, their checkpoints, and when a new one comes in.  Standard library only.

    python autoresearch/tools/keep3.py init --metric Q --split val --frames 873 --registry models/registry.json \
           --ckpt "runs/e123/{run}/model.pt" --card "runs/eval/{name}/{split}.json"      # once per problem; writes <records folder>/keep3.json
    python autoresearch/tools/keep3.py candidates [--top 12]       # every registry name with a card and a checkpoint: value, network, best name per network
    python autoresearch/tools/keep3.py consider NAME [--ledger ID | --by-hand "reason"] [--apply]
    python autoresearch/tools/keep3.py show [--hash]                # the three, their checkpoints (exist, size, sha256 with --hash)
    python autoresearch/tools/keep3.py --selftest

The rule (standing instruction of the person, 2026-11-01): a problem keeps its best three NETWORKS (one slot per checkpoint, not per registry name: the same
checkpoint with another assembly is one network). Their checkpoints stay complete; every other checkpoint may be slimmed or archived, never deleted
(hard rule 5). The list is a text record in git (<folder of the ledger>/keep3.json: value, card, commit, checkpoint path, size and sha256).
A candidate comes in when its value on the ranking card beats the weakest of the three by more than the band (the noise band, default 0.010), or
when fewer than three are held; on the same network it replaces the old name only by the band. A name without a checkpoint of its own (stored graphs,
a union of other networks) or whose file is missing is no candidate. Ranking is by the one metric only: the vertex budget (the card's vert100 is shown,
it is not the budget count of D25) and the cost (ms) are not gates here, so look at them in `candidates` before you take one in; a metric where lower is
better (project.json lower_is_better) is refused.
`consider` is a dry run. `--apply` needs the evidence: `--ledger ID` (the ledger rows ID or ID.* hold a decide.py verdict KEEP as the last verdict word of
the decision, and their text names the candidate or its run) or `--by-hand "reason"` (recorded as "by hand: ..."); the change is made under the lock of
the record, so two chats cannot lose each other's entry. Whenever the list changes, push (after `git fetch` and check_git.py): the printout names the
files to commit and the checkpoints that left (archive candidates; this tool moves nothing and commits nothing).
Ranking cards: <card pattern> with {name} and {split}; the card is a json with "metrics" {metric: value}, "split", "frames", "subset" and "commit"
(evaluation/evaluate.py of the lane problem). A problem whose results have another shape needs its own ranking rule first (bev_fusion: open).
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

VERDICT = re.compile(r"\b(KEEP|CUT|TIE|INCONCLUSIVE)\b")


def state_path():
    return os.path.join(os.path.dirname(P.path("ledger")), "keep3.json")


def load(path=None):
    path = path or state_path()
    if not os.path.exists(path):
        sys.exit(f"no {path}: `keep3.py init ...` first")
    return json.load(open(path, encoding="utf-8"))


def write(st, path):
    """atomic replace; the caller holds P.locked(path) when it read the state first"""
    P.atomic_write(path, json.dumps(st, indent=1, ensure_ascii=False) + "\n")


def registry_of(st, root):
    return {k: v for k, v in json.load(open(os.path.join(root, st["registry"]), encoding="utf-8-sig")).items() if isinstance(v, dict) and not k.startswith("_")}


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def info(st, root, reg, name):
    """(facts of one registry name on the ranking card, ''), or (None, why it is no candidate)"""
    if name not in reg:
        return None, f"{name} is not in the registry"
    run = reg[name].get("run")
    if not run:
        return None, f"{name} has no checkpoint of its own (stored graphs, or a union of other networks)"
    ck = st["ckpt"].format(run=run)
    if not os.path.exists(os.path.join(root, ck)):
        return None, f"the checkpoint {ck} of {name} does not exist (archived?): a network is kept with its weights"
    cp = os.path.join(root, st["card"].format(name=name, split=st["split"]))
    if not os.path.exists(cp):
        return None, f"no ranking card {os.path.relpath(cp, root)}"
    c = json.load(open(cp, encoding="utf-8-sig"))
    m = c.get("metrics", {})
    if st["metric"] not in m:
        return None, f"the card has no metric {st['metric']}"
    if c.get("split") != st["split"] or c.get("subset") or (st.get("frames") and c.get("frames") != st["frames"]):
        return None, f"the card is not the full {st['split']} split of {st.get('frames')} frames ({c.get('split')}, subset {c.get('subset')!r}, {c.get('frames')} frames)"
    return dict(name=name, key=run, runs=[run], value=float(m[st["metric"]]), vert100=m.get("vert100"), card=os.path.relpath(cp, root).replace(chr(92), "/"),
                commit=c.get("commit", ""), asm_ms=(c.get("info") or {}).get("asm_ms")), ""


def checkpoints(st, root, runs):
    out = []
    for r in runs:
        rel = st["ckpt"].format(run=r)
        f = os.path.join(root, rel)
        out.append(dict(path=rel, bytes=os.path.getsize(f), sha256=sha256(f)))
    return out


def decide(st, cand):
    """(action, text): action is one of enter / replace / out / same; text says why. Pure: the state is not changed"""
    ents, band = st["entries"], st["band"]
    same = [e for e in ents if e["key"] == cand["key"]]
    if same:
        e = same[0]
        if cand["value"] > e["value"] + band:
            return "replace", f"same network as {e['name']}: {cand['name']} is better by more than the band ({cand['value']:.4f} vs {e['value']:.4f}, band {band}), it takes the place"
        return "same", f"{e['name']} holds this network already ({cand['value']:.4f} vs {e['value']:.4f}, inside the band {band} or lower)"
    if len(ents) < st["slots"]:
        return "enter", f"{len(ents)} of {st['slots']} slots are taken: {cand['name']} ({cand['value']:.4f}) takes the next one"
    weak = min(ents, key=lambda e: e["value"])
    if cand["value"] > weak["value"] + band:
        return "enter", f"{cand['name']} ({cand['value']:.4f}) beats the weakest, {weak['name']} ({weak['value']:.4f}), by more than the band {band}: {weak['name']} leaves"
    return "out", f"{cand['name']} ({cand['value']:.4f}) does not beat the weakest, {weak['name']} ({weak['value']:.4f}), by more than the band {band}"


def evidence(a, cand):
    """the evidence text for --apply, or exit: a ledger KEEP that names the candidate, or a reason written by hand"""
    if a.by_hand and a.ledger:
        sys.exit("give --ledger or --by-hand, not both")
    if a.by_hand:
        return "by hand: " + a.by_hand
    if not a.ledger:
        sys.exit("--apply needs the evidence: --ledger ID (a KEEP verdict of decide.py that names the network) or --by-hand \"reason\"")
    rows = [r for r in csv.DictReader(open(P.path("ledger"), newline="", encoding="utf-8")) if r["id"] == a.ledger or r["id"].startswith(a.ledger + ".")]
    keep = [r for r in rows if (VERDICT.findall(r.get("decision") or "") or [""])[-1] == "KEEP"]
    if not keep:
        sys.exit(f"ledger id {a.ledger}: no row (or row of {a.ledger}.*) whose decision ends in the verdict KEEP ({len(rows)} rows found); run decide.py first, or use --by-hand \"reason\"")
    text = " ".join(" ".join(str(v) for v in r.values()) for r in keep).lower()
    if not any(re.search(r"(?<![\w.])" + re.escape(w.lower()) + r"(?!\w)", text) for w in [cand["name"], *cand["runs"]]):  # a whole name, not a piece of a word
        sys.exit(f"ledger id {a.ledger}: its KEEP rows name neither {cand['name']} nor {cand['runs']}: that experiment decided something else; name the right one, or use --by-hand \"reason\"")
    return f"ledger {a.ledger} KEEP"


def apply(st, cand, ck, action, ev, today=None):
    """change the state (returns the entries that left)"""
    ents, left = st["entries"], []
    entry = dict(name=cand["name"], key=cand["key"], runs=cand["runs"], value=cand["value"], vert100=cand["vert100"], asm_ms=cand["asm_ms"], card=cand["card"], commit=cand["commit"],
                 checkpoints=ck, since=today or dt.date.today().isoformat(), evidence=ev)
    if action == "replace":
        left = [e for e in ents if e["key"] == cand["key"]]
    elif action == "enter" and len(ents) >= st["slots"]:
        left = [min(ents, key=lambda e: e["value"])]
    for e in left:
        ents.remove(e)
        st.setdefault("left", []).append(dict(name=e["name"], key=e["key"], value=e["value"], left=entry["since"], replaced_by=cand["name"], checkpoints=[c["path"] for c in e["checkpoints"]]))
    ents.append(entry)
    ents.sort(key=lambda e: -e["value"])
    return left


def kept_paths(st):
    return {c["path"] for e in st["entries"] for c in e["checkpoints"]}


def cmd_init(a, root):
    path = state_path()
    if os.path.exists(path):
        sys.exit(f"{path} exists: change it with `consider ... --apply`")
    if a.metric in P.cfg().get("lower_is_better", []):
        sys.exit(f"{a.metric} is a metric where lower is better (project.json lower_is_better): the ranking would put the worst network first")
    st = dict(about="the best three networks of this problem (autoresearch/tools/keep3.py); one slot per checkpoint; a text record in git", metric=a.metric, split=a.split,
              frames=a.frames, slots=a.slots, band=a.band, registry=a.registry, ckpt=a.ckpt, card=a.card, entries=[], left=[])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with P.locked(path):
        write(st, path)
    print(f"wrote {os.path.relpath(path, root)} (empty, {a.slots} slots, ranked by {a.metric} on {a.split})")


def cmd_candidates(a, root):
    st = load()
    reg, rows, skipped = registry_of(st, root), [], {}
    for n in reg:
        i, why = info(st, root, reg, n)
        if i:
            rows.append(i)
        else:
            skipped[n] = why
    best = {}
    for i in rows:
        if i["key"] not in best or i["value"] > best[i["key"]]["value"]:
            best[i["key"]] = i
    kept = {e["key"] for e in st["entries"]}
    print(f"{st['metric']} on {st['split']} ({st.get('frames')} frames), best name per network, {len(rows)} names are candidates, {len(best)} networks ({len(skipped)} names are not: no checkpoint of their own / "
          "missing file / no full-split card); look at vert100 and the cost before you take one in:")
    for i in sorted(best.values(), key=lambda i: -i["value"])[:a.top]:
        print(f"  {i['value']:.4f}  {i['name']:22} {i['key'][:46]:46} vert100 {i['vert100'] if i['vert100'] is None else round(i['vert100'], 1)}{'  <- kept' if i['key'] in kept else ''}")


def cmd_consider(a, root):
    sp = state_path()
    st = load()
    reg = registry_of(st, root)
    cand, why = info(st, root, reg, a.name)
    if not cand:
        sys.exit(f"{a.name}: {why}")
    action, text = decide(st, cand)
    print(f"{action.upper()}: {text}")
    if not a.apply:
        return
    if action in ("same", "out"):
        print("nothing to apply")
        return
    ev = evidence(a, cand)
    ck = checkpoints(st, root, cand["runs"])        # hashed once, before the lock: it can take seconds
    with P.locked(sp):
        st = load()                                  # the state as it is now: another chat may have changed it while we hashed
        action, text = decide(st, cand)
        if action in ("same", "out"):
            sys.exit(f"changed meanwhile, nothing applied: {action.upper()}: {text}")
        left = apply(st, cand, ck, action, ev)
        write(st, sp)
    gone = sorted({c["path"] for e in left for c in e["checkpoints"]} - kept_paths(st))
    print(f"applied ({ev}): {[e['name'] for e in st['entries']]}")
    print("next: commit the text record (own paths, check_git.py --paths ...) and push it (git fetch, check_git.py):")
    print(f"  {os.path.relpath(sp, root)}  {cand['card']}  (+ the registry entry, the ledger row, the run record)")
    if gone:
        print("left the three (slim or archive these checkpoints, never delete; this tool moves nothing): " + ", ".join(gone))
    return gone


def cmd_show(a, root):
    st = load()
    print(f"best {st['slots']} by {st['metric']} on {st['split']} (band {st['band']}), {len(st['entries'])} held:")
    bad = 0
    for e in st["entries"]:
        print(f"  {e['value']:.4f}  {e['name']:20} since {e['since']}  {e['evidence']}")
        for c in e["checkpoints"]:
            f = os.path.join(root, c["path"])
            st_ = ("MISSING" if not os.path.exists(f) else "size differs" if os.path.getsize(f) != c["bytes"]
                   else "SHA256 DIFFERS" if a.hash and sha256(f) != c["sha256"] else "ok")
            bad += st_ != "ok"
            print(f"          {c['path']}  {c['bytes'] / 1048576:.0f} MB  {st_}")
    if st.get("left"):
        print("left: " + ", ".join(f"{x['name']} ({x['left']})" for x in st["left"]))
    return 1 if bad else 0


def selftest():
    T = tempfile.mkdtemp(prefix="ar_keep3_")
    old = os.environ.get("AR_PROJECT")
    try:
        os.makedirs(f"{T}/records")
        os.makedirs(f"{T}/runs/ck")
        os.makedirs(f"{T}/cards")
        json.dump(dict(ledger="records/LEDGER.csv", ledger_md="records/LEDGER.md", lower_is_better=["vert100"]), open(f"{T}/project.json", "w"))
        reg = {"_about": "x", **{n: dict(kind="builder", run=r) for n, r in (("a", "ra"), ("a_dp", "ra"), ("b", "rb"), ("c", "rc"), ("d", "rd"), ("e", "re"), ("g", "rg"), ("w", "re"), ("m", "rm"))},
               "u": dict(kind="union", members=["a", "b"]), "gr": dict(kind="graphs")}
        json.dump(reg, open(f"{T}/registry.json", "w"))
        for r in ("ra", "rb", "rc", "rd", "re"):
            open(f"{T}/runs/ck/{r}.pt", "wb").write(r.encode() * 100)

        def card(n, q, frames=100, split="val", subset=""):
            os.makedirs(f"{T}/cards/{n}", exist_ok=True)
            json.dump(dict(name=n, split=split, subset=subset, frames=frames, commit="abc", metrics=dict(Q=q, vert100=12.0), info=dict(asm_ms=9.0)), open(f"{T}/cards/{n}/val.json", "w"))
        for n, q in (("a", 0.70), ("a_dp", 0.705), ("b", 0.72), ("c", 0.65), ("d", 0.725), ("e", 0.60), ("u", 0.75)):
            card(n, q)
        card("g", 0.9, frames=50)           # not the full split (and no checkpoint file)
        card("w", 0.9, split="test")        # a card of another split under the val name
        card("m", 0.9, subset="night")      # a subset card (no checkpoint file either, so the card test below uses e)
        os.environ["AR_PROJECT"] = T
        with open(f"{T}/records/LEDGER.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "decision", "hypothesis"])
            w.writeheader()
            w.writerows([dict(id="E1.a", decision="decide.py 2026-11-01: CUT (effect -0.01)", hypothesis="rd and rb"), dict(id="E3.a", decision="decide.py: TIE first, then KEEP (effect +0.02)", hypothesis="the rd network"),
                         dict(id="E10", decision="KEEP", hypothesis="rd rb ra"), dict(id="E4", decision="KEEP then CUT", hypothesis="rd")])
        ns = lambda **k: argparse.Namespace(**{**dict(metric="Q", split="val", frames=100, slots=3, band=0.01, registry="registry.json", ckpt="runs/ck/{run}.pt", card="cards/{name}/{split}.json",
                                                      hash=False, top=10, apply=False, ledger="", by_hand=""), **k})
        refused = lambda **k: _raises(lambda: cmd_consider(ns(**k), T))
        _raises(lambda: cmd_init(ns(metric="vert100"), T))   # lower is better: refused, nothing written
        assert not os.path.exists(state_path())
        cmd_init(ns(), T)
        st = load()
        rg = registry_of(st, T)
        # no candidates: stored graphs / union (no checkpoint of its own), a half-size card, another split, a subset, a missing file; the dry run says so too
        for n in ("g", "gr", "u", "w", "m"):
            assert info(st, T, rg, n)[0] is None, n
        assert "full" in info(st, T, rg, "w")[1] and "does not exist" in info(st, T, rg, "g")[1]
        assert "no checkpoint of its own" in info(st, T, rg, "gr")[1] and "no checkpoint of its own" in info(st, T, rg, "u")[1]
        card("m", 0.9, subset="night")
        open(f"{T}/runs/ck/rm.pt", "wb").write(b"m")
        assert "full" in info(st, T, rg, "m")[1] and refused(name="gr") and refused(name="u")
        c_a, c_adp = (info(st, T, rg, n)[0] for n in ("a", "a_dp"))
        assert c_a["key"] == c_adp["key"] == "ra"
        # the first three come in without a comparison
        for n in ("a", "b", "c"):
            cmd_consider(ns(name=n, apply=True, by_hand="seed"), T)
        assert [e["name"] for e in load()["entries"]] == ["b", "a", "c"] and load()["entries"][0]["evidence"] == "by hand: seed"
        # the band: the weakest is c 0.65; +0.005 stays out, +0.011 comes in (the boundary itself is inside the band)
        card("e", 0.655)
        assert decide(load(), info(st, T, rg, "e")[0])[0] == "out"
        card("e", 0.6599)
        assert decide(load(), info(st, T, rg, "e")[0])[0] == "out"
        card("e", 0.6601)
        assert decide(load(), info(st, T, rg, "e")[0])[0] == "enter"
        card("e", 0.60)
        # exactly the weakest plus the band is not MORE than the band (binary-exact numbers); with a free slot anything comes in
        full = dict(entries=[dict(name="w", key="kw", value=1.0)], slots=1, band=0.5)
        assert decide(full, dict(name="x", key="kx", value=1.5))[0] == "out" and decide(full, dict(name="x", key="kx", value=1.5001))[0] == "enter"
        assert decide(dict(full, slots=2), dict(name="x", key="kx", value=0.0))[0] == "enter"
        cmd_candidates(ns(), T)
        # same network: a_dp replaces a only by more than the band (0.705 vs 0.70: no)
        assert decide(load(), c_adp)[0] == "same"
        # a dry run and an apply of an 'out' candidate leave the file as it was, byte for byte
        raw = open(state_path(), "rb").read()
        cmd_consider(ns(name="e"), T)
        cmd_consider(ns(name="e", apply=True, by_hand="x"), T)
        assert open(state_path(), "rb").read() == raw, "dry run / out must not write"
        cmd_consider(ns(name="d"), T)
        assert open(state_path(), "rb").read() == raw, "consider without --apply must not write"
        # the evidence: no KEEP verdict as the LAST verdict word (CUT, 'KEEP then CUT'), an id that only prefixes another (E1 vs E10), a KEEP that names another
        # network, both --ledger and --by-hand, nothing: all refused and nothing changes
        for bad in (dict(ledger="E1"), dict(ledger="E99"), dict(ledger="E4"), dict(ledger="E10", by_hand="x"), dict()):
            assert refused(name="d", apply=True, **bad), bad
        card("e", 0.6601)
        assert refused(name="e", apply=True, ledger="E3"), "E3's KEEP names rd, not e / re"
        card("e", 0.60)
        assert open(state_path(), "rb").read() == raw and len(load()["entries"]) == 3, "a refused apply changes nothing"
        gone = cmd_consider(ns(name="d", apply=True, ledger="E3"), T)    # 'TIE first, then KEEP': the last verdict word counts
        assert gone == ["runs/ck/rc.pt"], gone
        s2 = load()
        assert [e["name"] for e in s2["entries"]] == ["d", "b", "a"] and s2["left"][0]["name"] == "c" and s2["entries"][0]["evidence"] == "ledger E3 KEEP", s2
        assert kept_paths(s2) == {"runs/ck/rd.pt", "runs/ck/rb.pt", "runs/ck/ra.pt"} and "runs/ck/rc.pt" not in kept_paths(s2)
        assert s2["entries"][0]["checkpoints"][0]["sha256"] == sha256(f"{T}/runs/ck/rd.pt")
        # a replace: the same network with a better assembly takes the place, the old name goes to 'left', its checkpoint is still held (nothing to archive)
        card("a_dp", 0.72)
        assert decide(load(), info(st, T, rg, "a_dp")[0])[0] == "replace"
        assert cmd_consider(ns(name="a_dp", apply=True, ledger="E10"), T) == [], "the replaced name's checkpoint is still held: nothing to archive"
        s3 = load()
        assert {e["name"] for e in s3["entries"]} == {"d", "a_dp", "b"} and len(s3["entries"]) == 3, s3["entries"]
        assert s3["left"][-1]["name"] == "a" and kept_paths(s3) == kept_paths(s2)
        # two chats at once: while our apply hashes its checkpoint, another chat takes a network in; ours must be decided on the state as it is then, so
        # both are kept (a stale state would drop the other chat's entry)
        open(f"{T}/runs/ck/rg.pt", "wb").write(b"g" * 50)
        card("g", 0.79)
        card("e", 0.80)
        real = checkpoints

        def racing(st_, root_, runs_):
            out = real(st_, root_, runs_)
            globals()["checkpoints"] = real
            cmd_consider(ns(name="g", apply=True, by_hand="the other chat"), T)
            return out
        globals()["checkpoints"] = racing
        try:
            cmd_consider(ns(name="e", apply=True, by_hand="this chat"), T)
        finally:
            globals()["checkpoints"] = real
        assert [e["name"] for e in load()["entries"]] == ["e", "g", "d"], [e["name"] for e in load()["entries"]]
        assert not os.path.exists(state_path() + ".lock")
        card("e", 0.60)
        # show: ok, then a changed file is seen (size) and a changed byte only with --hash
        assert cmd_show(ns(), T) == 0
        open(f"{T}/runs/ck/rd.pt", "wb").write(b"r" * 199 + b"X")  # 200 bytes like the original "rd" * 100
        assert cmd_show(ns(), T) == 0 and cmd_show(ns(hash=True), T) == 1, "same size, other bytes: only the hash sees it"
        open(f"{T}/runs/ck/rd.pt", "wb").write(b"short")
        assert cmd_show(ns(), T) == 1
        print("keep3 selftest passed")
    finally:
        if old is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old
        shutil.rmtree(T, ignore_errors=True)


def _raises(fn):
    """True when fn exits (SystemExit) - a refusal"""
    try:
        fn()
    except SystemExit:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    i = sub.add_parser("init")
    i.add_argument("--metric", default="Q")
    i.add_argument("--split", default="val")
    i.add_argument("--frames", type=int, default=0)
    i.add_argument("--slots", type=int, default=3)
    i.add_argument("--band", type=float, default=0.010)
    i.add_argument("--registry", required=True)
    i.add_argument("--ckpt", required=True, help="checkpoint path per run, relative to the problem folder, with {run}")
    i.add_argument("--card", required=True, help="ranking card per registry name, with {name} and {split}")
    c = sub.add_parser("candidates")
    c.add_argument("--top", type=int, default=12)
    k = sub.add_parser("consider")
    k.add_argument("name")
    k.add_argument("--ledger", default="")
    k.add_argument("--by-hand", default="")
    k.add_argument("--apply", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command (init, candidates, consider, show) or --selftest")
    return {"init": cmd_init, "candidates": cmd_candidates, "consider": cmd_consider, "show": cmd_show}[a.cmd](a, P.root())


if __name__ == "__main__":
    sys.exit(main())
