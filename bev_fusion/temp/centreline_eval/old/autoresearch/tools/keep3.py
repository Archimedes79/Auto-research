"""The best three networks of a problem: which ones, their checkpoints, and when a new one comes in.  Standard library only.

    python autoresearch/tools/keep3.py init --metric Q --split val --frames 873 --registry models/registry.json \
           --ckpt "runs/e123/{run}/model.pt" --card "runs/eval/{name}/{split}.json"      # once per problem; writes <records folder>/keep3.json
    python autoresearch/tools/keep3.py candidates [--top 12]       # every registry name with a card and a checkpoint: value, network, best name per network
    python autoresearch/tools/keep3.py consider NAME [--ledger ID | --by-hand "reason"] [--apply]
    python autoresearch/tools/keep3.py show [--hash]                # the three, their checkpoints (exist, size, sha256 with --hash)
    python autoresearch/tools/keep3.py pin NAME --reason "..."      # a pinned network (the champion: a protected registry entry) is never pushed out by a better newcomer
    python autoresearch/tools/keep3.py unpin NAME --reason "..."
    python autoresearch/tools/keep3.py --selftest

The rule (standing instruction of the person, 2026-11-01): a problem keeps its best three NETWORKS (one slot per checkpoint, not per registry name: the same
checkpoint with another assembly is one network). Their checkpoints stay complete; every other checkpoint may be slimmed or archived, never deleted
(hard rule 5). The list is a text record in git (<folder of the ledger>/keep3.json: value, card, commit, checkpoint path, size and sha256).
A candidate comes in when its value on the ranking card beats the weakest of the three by more than the band (the noise band, default 0.010), or
when fewer than three are held; on the same network it replaces the old name only by the band. The one that leaves is the weakest UNPINNED entry: a pinned
entry (`pin`, with a reason; keep the protected champion pinned, a better newcomer must not evict it) stays whatever the ranking says, and when all are pinned nobody comes in. A name without a checkpoint of its own (stored graphs,
a union of other networks) or whose file is missing is no candidate. Ranking is by the one metric only: the vertex budget (the card's vert100 is shown,
it is not the budget count of D25) and the cost (ms) are not gates here, so look at them in `candidates` before you take one in; a metric where lower is
better (project.json lower_is_better) is refused.
`consider` is a dry run. `--apply` needs the evidence: `--ledger ID` (the ledger rows ID or ID.* hold a decide.py verdict KEEP as the last verdict word of
the decision, and their text names the candidate or its run) or `--by-hand "reason"` (recorded as "by hand: ..."); the change is made under the lock of
the record, so two chats cannot lose each other's entry. Whenever the list changes, push (after `git fetch` and check_git.py): the printout names the
files to commit and the checkpoints that left (archive candidates; this tool moves nothing and commits nothing).
Ranking cards: <card pattern> with {name} and {split}; the card is a json with "metrics" {metric: value}, "split", "frames", "subset" and "commit"
(evaluation/evaluate.py of the lane problem).
ARMS (bev_fusion, since 2026-11-01): `init --source arms --metric board7 --metric-keys k1,k2,... --gt-signature S --band 0.02 --ckpt "runs/{run}/ckpt_*.pt"
--card "runs/{name}/eval/metrics.json"`: a candidate is an ARM (runs/<arm>_s0, _s1, ... are one network family; a run without a seed suffix is its own arm), its
value is the mean of the given keys of the flat metrics.json of each seed run (all keys higher-is-better), averaged over the seeds, and every seed of the arm must be
evaluated on the ranking split and GT signature (another GT is another question); the NEWEST checkpoint of each seed run is kept, so an arm holds all its seed checkpoints.
For bev_fusion the keys are the six map IoUs and the mAP of evaluation/board.py (a composite chosen by convention: it is a ranking, not a verdict; the card's rule decides).
"""
import argparse
import contextlib
import csv
import datetime as dt
import glob
import hashlib
import io
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


def arm_of(run):
    """the arm a run belongs to: t8_ctl_s1 -> t8_ctl (seed runs are one network family), a run without a seed suffix is its own arm"""
    return re.sub(r"_s\d+$", "", run)


def registry_of(st, root):
    if st.get("source") == "arms":                    # no registry: every arm with an evaluated run folder (runs/<arm>_s<N> or runs/<arm>)
        arms, (pre, post) = {}, st["card"].split("{name}")
        dirs = sorted(glob.glob(os.path.join(root, pre, "*")))
        for d in dirs:
            if os.path.isfile(os.path.join(d, post.format(split=st["split"]).strip("/\\"))):
                run = os.path.basename(d)
                arms.setdefault(arm_of(run), {"run": arm_of(run), "runs": []})["runs"].append(run)
        for d in dirs:                                 # a seed run of an arm that is not evaluated yet is listed too: the arm then names it as the reason it is no candidate
            run = os.path.basename(d)
            if os.path.isdir(d) and arm_of(run) in arms and run not in arms[arm_of(run)]["runs"] and run != arm_of(run):
                arms[arm_of(run)]["runs"].append(run)
        return arms
    return {k: v for k, v in json.load(open(os.path.join(root, st["registry"]), encoding="utf-8-sig")).items() if isinstance(v, dict) and not k.startswith("_")}


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_file(st, root, run, prefer=None):
    """the checkpoint path (relative to the problem folder) of a run: the pattern as it is, or with a * the match named `prefer` (the file a card was made
    from) when there is one, else the newest (highest number in the name); None when absent"""
    rel = st["ckpt"].format(run=run)
    if "*" not in rel:
        return rel if os.path.exists(os.path.join(root, rel)) else None
    hits = glob.glob(os.path.join(root, rel))
    named = [h for h in hits if prefer and os.path.basename(h) == prefer]
    if named:
        return os.path.relpath(named[0], root).replace(chr(92), "/")
    num = lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]) if re.findall(r"\d+", os.path.basename(f)) else -1  # noqa: E731
    return os.path.relpath(max(hits, key=num), root).replace(chr(92), "/") if hits else None


def info_arm(st, root, reg, name):
    """an ARM (all its seed runs): the value is the mean over the seeds of the mean of st['metric_keys'] in each run's metrics.json (a flat card
    {"split", "gt_signature", "metrics": {...}}); every seed must carry every key, the split and the GT signature of the ranking, and a checkpoint"""
    runs, vals, cards, files = reg[name]["runs"], [], [], []
    for r in runs:
        cp = os.path.join(root, st["card"].format(name=r, split=st["split"]))
        if not os.path.exists(cp):
            return None, f"seed run {r} of the arm has no evaluation ({os.path.relpath(cp, root)}): every seed must be evaluated"
        c = json.load(open(cp, encoding="utf-8-sig"))
        ck = ckpt_file(st, root, r, prefer=c.get("checkpoint"))
        if not ck:
            return None, f"the checkpoint of {r} does not exist (archived?): a network is kept with its weights"
        if c.get("checkpoint") and os.path.basename(ck) != c["checkpoint"]:
            return None, f"{r} was evaluated on {c['checkpoint']}, which does not exist (only {os.path.basename(ck)}): the weights must be the evaluated ones"
        files.append(ck)
        m = c.get("metrics", {})
        miss = [k for k in st["metric_keys"] if not isinstance(m.get(k), (int, float)) or m[k] != m[k] or abs(m[k]) == float("inf")]
        if miss:
            return None, f"{r} has no finite value for {len(miss)} of the {len(st['metric_keys'])} ranking keys (e.g. {miss[0]})"
        if c.get("split") != st["split"] or (st.get("gt_signature") and c.get("gt_signature") != st["gt_signature"]):
            return None, f"{r} was not evaluated on the ranking question ({c.get('split')}, GT {c.get('gt_signature')} against {st['split']}, GT {st.get('gt_signature')})"
        vals.append(sum(float(m[k]) for k in st["metric_keys"]) / len(st["metric_keys"]))
        cards.append(os.path.relpath(cp, root).replace(chr(92), "/"))
    order = sorted(range(len(runs)), key=lambda i: runs[i])
    return dict(name=name, key=name, runs=[runs[i] for i in order], ckpts=[files[i] for i in order], value=sum(vals) / len(vals), vert100=None, card=cards[0], commit="", asm_ms=None), ""


def info(st, root, reg, name):
    """(facts of one registry name on the ranking card, ''), or (None, why it is no candidate)"""
    if name not in reg:
        return None, f"{name} is not in the registry"
    if st.get("source") == "arms":
        return info_arm(st, root, reg, name)
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


def checkpoints(st, root, runs, files=None):
    out = []
    for i, r in enumerate(runs):
        rel = files[i] if files else ckpt_file(st, root, r)
        f = os.path.join(root, rel)
        out.append(dict(path=rel, bytes=os.path.getsize(f), sha256=sha256(f)))
    return out


def weakest(ents):
    """the entry a newcomer would push out: the weakest one that is not pinned (None when every entry is pinned)"""
    free = [e for e in ents if not e.get("pinned")]
    return min(free, key=lambda e: e["value"]) if free else None


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
    weak = weakest(ents)
    if weak is None:
        return "out", f"every one of the {len(ents)} slots is pinned ({', '.join(e['name'] for e in ents)}): nobody can leave, {cand['name']} stays out (`unpin` first, with a reason)"
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
        pins = [e["pinned"] for e in left if e.get("pinned")]
        if pins:                                       # the same network under a better assembly: the pin stays with the network
            entry["pinned"] = pins[0]
    elif action == "enter" and len(ents) >= st["slots"]:
        w = weakest(ents)
        if w is None:
            sys.exit("every slot is pinned: nothing can leave")
        left = [w]
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
    keys = [k for k in a.metric_keys.split(",") if k]
    low = set(P.cfg().get("lower_is_better", []))
    if a.metric in low or any(k in low or k.rsplit("/", 1)[-1] in low for k in keys):
        sys.exit(f"{a.metric} or one of its keys is a metric where lower is better (project.json lower_is_better): the ranking would put the worst network first")
    if bool(keys) != (a.source == "arms") or (a.source != "arms" and not a.registry):
        sys.exit("--source arms needs --metric-keys (the value is their mean over the seeds); a registry ranking needs --registry and no --metric-keys")
    st = dict(about="the best three networks of this problem (autoresearch/tools/keep3.py); one slot per checkpoint (per arm with all its seeds); a text record in git", metric=a.metric,
              split=a.split, frames=a.frames, slots=a.slots, band=a.band, registry=a.registry, ckpt=a.ckpt, card=a.card, entries=[], left=[])
    if a.source == "arms":
        st.update(source="arms", metric_keys=keys, gt_signature=a.gt_signature)
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
    ck = checkpoints(st, root, cand["runs"], cand.get("ckpts"))        # hashed once, before the lock: it can take seconds
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


def cmd_pin(a, root):
    """pin / unpin a held network: a pinned one (e.g. the champion, a protected registry entry) never leaves because a newcomer is better"""
    if not a.reason.strip():
        sys.exit("--reason is needed (why this network must stay, e.g. 'protected registry entry: the champion')")
    sp = state_path()
    with P.locked(sp):
        st = load()
        hits = [e for e in st["entries"] if e["name"] == a.name]
        if not hits:
            sys.exit(f"{a.name} is not held; held: {[e['name'] for e in st['entries']]}")
        if a.cmd == "pin":
            hits[0]["pinned"] = a.reason.strip()
        elif not hits[0].get("pinned"):
            sys.exit(f"{a.name} is not pinned")
        else:
            del hits[0]["pinned"]
        st.setdefault("pin_log", []).append(dict(name=a.name, action=a.cmd, reason=a.reason.strip(), date=dt.date.today().isoformat()))
        write(st, sp)
    print(f"{a.name}: {a.cmd}ned ({a.reason.strip()}); commit {os.path.relpath(sp, root)} (own path, check_git.py --paths), push after git fetch and check_git.py")


def cmd_show(a, root):
    st = load()
    print(f"best {st['slots']} by {st['metric']} on {st['split']} (band {st['band']}), {len(st['entries'])} held:")
    bad = 0
    for e in st["entries"]:
        print(f"  {e['value']:.4f}  {e['name']:20} since {e['since']}  {e['evidence']}" + (f"  PINNED: {e['pinned']}" if e.get("pinned") else ""))
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
                                                      hash=False, top=10, apply=False, ledger="", by_hand="", source="registry", metric_keys="", gt_signature=""), **k})
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

        def racing(st_, root_, runs_, files_=None):
            out = real(st_, root_, runs_, files_)
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
        # the command line end to end: an apply returns a list (the checkpoints that left), the exit code must still be 0
        card("e", 0.95)
        old_argv, sys.argv = sys.argv, ["keep3.py", "consider", "e", "--apply", "--by-hand", "selftest"]
        try:
            assert main() == 0 and load()["entries"][0]["name"] == "e" and load()["entries"][0]["value"] == 0.95
        finally:
            sys.argv = old_argv
        card("e", 0.60)
        # pins: a better newcomer pushes out the weakest UNPINNED entry, never the pinned champion; all pinned = nobody comes in; a replace keeps the pin
        pst = dict(entries=[dict(name="champ", key="k1", value=0.70, pinned="protected", checkpoints=[]), dict(name="x", key="k2", value=0.74, checkpoints=[]),
                            dict(name="y", key="k3", value=0.75, checkpoints=[])], slots=3, band=0.01)
        newcomer = dict(name="n", key="k9", runs=["rn"], value=0.80, vert100=1, asm_ms=1, card="c", commit="c")
        act, txt = decide(pst, newcomer)
        assert act == "enter" and "beats the weakest, x " in txt, (act, txt)
        left = apply(pst, newcomer, [], "enter", "test")
        assert [e["name"] for e in left] == ["x"] and {e["name"] for e in pst["entries"]} == {"champ", "y", "n"}, (left, pst["entries"])
        allpin = dict(entries=[dict(name=f"p{i}", key=f"kp{i}", value=0.6 + i / 100, pinned="r", checkpoints=[]) for i in range(3)], slots=3, band=0.01)
        act, txt = decide(allpin, newcomer)
        assert act == "out" and "pinned" in txt, (act, txt)
        assert _raises(lambda: apply(allpin, newcomer, [], "enter", "test")), "an apply with every slot pinned must refuse"
        samekey = dict(entries=[dict(name="champ", key="ka", value=0.70, pinned="protected", checkpoints=[])], slots=3, band=0.01)
        better = dict(newcomer, name="champ2", key="ka", value=0.80)
        assert decide(samekey, better)[0] == "replace"
        apply(samekey, better, [], "replace", "test")
        assert samekey["entries"][0]["name"] == "champ2" and samekey["entries"][0].get("pinned") == "protected", samekey
        # the command: a reason is needed, an unknown name and an unpin of an unpinned one are refused, the log keeps who/why, show names the pin
        assert _raises(lambda: cmd_pin(ns(cmd="pin", name="d", reason="  "), T)) and _raises(lambda: cmd_pin(ns(cmd="pin", name="zz", reason="r"), T))
        assert _raises(lambda: cmd_pin(ns(cmd="unpin", name="d", reason="r"), T)), "d is not pinned yet"
        cmd_pin(ns(cmd="pin", name="d", reason="protected registry entry"), T)
        held = {e["name"]: e for e in load()["entries"]}
        assert held["d"]["pinned"] == "protected registry entry" and not held["e"].get("pinned") and load()["pin_log"][-1]["action"] == "pin"
        buf_ = io.StringIO()
        with contextlib.redirect_stdout(buf_):
            cmd_show(ns(), T)
        assert "PINNED: protected registry entry" in buf_.getvalue(), buf_.getvalue()
        weak_name = min(load()["entries"], key=lambda e: e["value"])["name"]
        assert weak_name == "d" and weakest(load()["entries"])["name"] != "d", "the weakest is d, but d is pinned: another one is the one to leave"
        cmd_pin(ns(cmd="unpin", name="d", reason="no longer the champion"), T)
        assert not {e["name"]: e for e in load()["entries"]}["d"].get("pinned") and len(load()["pin_log"]) == 2
        # show: ok, then a changed file is seen (size) and a changed byte only with --hash
        assert cmd_show(ns(), T) == 0
        open(f"{T}/runs/ck/rd.pt", "wb").write(b"r" * 199 + b"X")  # 200 bytes like the original "rd" * 100
        assert cmd_show(ns(), T) == 0 and cmd_show(ns(hash=True), T) == 1, "same size, other bytes: only the hash sees it"
        open(f"{T}/runs/ck/rd.pt", "wb").write(b"short")
        assert cmd_show(ns(), T) == 1
        # an arms project (the bev_fusion shape): a flat metrics.json per seed run, the newest checkpoint of a glob, value = mean of the keys, then the mean over the seeds
        T2 = tempfile.mkdtemp(prefix="ar_keep3b_")
        try:
            os.environ["AR_PROJECT"] = T2
            os.makedirs(f"{T2}/records")
            json.dump(dict(ledger="records/LEDGER.csv", ledger_md="records/LEDGER.md", lower_is_better=["ave"]), open(f"{T2}/project.json", "w"))

            def arm_run(run, a_, b_, sig="S1", split="val", omit=None, ck=None):
                os.makedirs(f"{T2}/runs/{run}/eval")
                for s in (1000, 3000):
                    open(f"{T2}/runs/{run}/ckpt_{s:08d}.pt", "wb").write(f"{run}{s}".encode())
                m = {"m/a": a_, "m/b": b_}
                if omit:
                    del m[omit]
                json.dump(dict(split=split, gt_signature=sig, metrics=m, **({"checkpoint": ck} if ck else {})), open(f"{T2}/runs/{run}/eval/metrics.json", "w"))
            for r, v in (("x_s0", .5), ("x_s1", .6), ("y_s0", .7), ("z", .3)):
                arm_run(r, v, v)
            arm_run("o_s0", .9, .9, sig="S2")
            arm_run("p_s0", .9, .9, omit="m/b")
            arm_run("q_s0", .9, .9, split="test")
            arm_run("w_s0", .4, .4, ck="ckpt_00001000.pt")       # evaluated on an OLDER checkpoint than the newest one: that file is the network
            arm_run("v_s0", .4, .4, ck="ckpt_00002000.pt")       # evaluated on a checkpoint that does not exist any more
            arm_run("u_s0", .4, .4)
            os.makedirs(f"{T2}/runs/u_s1")                       # a seed that is trained but not evaluated
            arms = dict(source="arms", metric="board", metric_keys="m/a,m/b", registry="", ckpt="runs/{run}/ckpt_*.pt", card="runs/{name}/eval/metrics.json", gt_signature="S1", frames=0, band=0.02)
            _raises(lambda: cmd_init(ns(**{**arms, "metric_keys": "m/a,ave"}), T2))      # a lower-is-better key: refused
            _raises(lambda: cmd_init(ns(**{**arms, "metric_keys": ""}), T2))             # arms without keys: refused
            _raises(lambda: cmd_init(ns(**{**arms, "source": "registry", "registry": ""}), T2))
            assert not os.path.exists(f"{T2}/records/keep3.json")
            cmd_init(ns(**arms), T2)
            st2 = load()
            rg2 = registry_of(st2, T2)
            assert sorted(rg2) == ["o", "p", "q", "u", "v", "w", "x", "y", "z"] and sorted(rg2["x"]["runs"]) == ["x_s0", "x_s1"], sorted(rg2)
            ix = info(st2, T2, rg2, "x")[0]
            assert abs(ix["value"] - 0.55) < 1e-12 and ix["runs"] == ["x_s0", "x_s1"] and ix["key"] == "x", ix
            assert checkpoints(st2, T2, ["x_s0"])[0]["path"] == "runs/x_s0/ckpt_00003000.pt", "the newest checkpoint of the glob"
            iw = info(st2, T2, rg2, "w")[0]
            assert iw["ckpts"] == ["runs/w_s0/ckpt_00001000.pt"] and checkpoints(st2, T2, iw["runs"], iw["ckpts"])[0]["path"] == "runs/w_s0/ckpt_00001000.pt", iw
            assert "does not exist" in info(st2, T2, rg2, "v")[1] or "evaluated on" in info(st2, T2, rg2, "v")[1]
            assert info(st2, T2, rg2, "v")[0] is None and "every seed must be evaluated" in info(st2, T2, rg2, "u")[1], info(st2, T2, rg2, "u")
            assert "ranking question" in info(st2, T2, rg2, "o")[1] and "ranking question" in info(st2, T2, rg2, "q")[1] and "finite value" in info(st2, T2, rg2, "p")[1]
            for n in ("x", "y", "z"):
                cmd_consider(ns(**{**arms, "name": n, "apply": True, "by_hand": "selftest"}), T2)
            held = {e["name"]: e for e in load()["entries"]}
            assert sorted(held) == ["x", "y", "z"] and [c["path"] for c in held["x"]["checkpoints"]] == ["runs/x_s0/ckpt_00003000.pt", "runs/x_s1/ckpt_00003000.pt"]
            assert cmd_show(ns(**arms), T2) == 0
            os.remove(f"{T2}/runs/x_s1/ckpt_00003000.pt")
            assert cmd_show(ns(**arms), T2) == 1, "a lost seed checkpoint of a held arm is seen"
        finally:
            shutil.rmtree(T2, ignore_errors=True)
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
    i.add_argument("--registry", default="")
    i.add_argument("--source", choices=("registry", "arms"), default="registry", help="arms: no registry, every arm of the run folders (seed runs are one family); needs --metric-keys")
    i.add_argument("--metric-keys", default="", help="arms: comma separated keys of the flat metrics.json; the value is their mean (all higher-is-better), then the mean over the seeds")
    i.add_argument("--gt-signature", default="", help="arms: only runs evaluated on this GT signature are candidates")
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
    for verb in ("pin", "unpin"):
        pn = sub.add_parser(verb, help="a pinned network (the champion: a protected registry entry) never leaves because a newcomer is better" if verb == "pin" else "remove the pin")
        pn.add_argument("name")
        pn.add_argument("--reason", required=True)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command (init, candidates, consider, show, pin, unpin) or --selftest")
    rc = {"init": cmd_init, "candidates": cmd_candidates, "consider": cmd_consider, "show": cmd_show, "pin": cmd_pin, "unpin": cmd_pin}[a.cmd](a, P.root())
    return rc if isinstance(rc, int) else 0  # consider returns the checkpoints that left (a list), which sys.exit would print and turn into exit code 1


if __name__ == "__main__":
    sys.exit(main())
