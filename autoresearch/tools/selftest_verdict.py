"""Self-test of the verdict path (decide.py, experiment.py, keep3.py, metrics.py, plan.py) in a throw-away problem folder; no real ledger is touched.

    python autoresearch/tools/selftest_verdict.py        (selftest.py runs it at its end; exit 0 = all sections passed)

One section per rule, each with a known answer: the rule fixed before the run (rule_sha, card_for_run, the gate of decide.py: ledger status ok, card_sha, --by-hand),
the decision cell (token first, both orders read by keep3 / metrics / plan, a 600-character prose), the card checks (at_least, subsets, min_effect / max_loss, a card file
listed twice, sources), the order and the failures of --write (ledger first, atomic files, a warning instead of an abort), the typed form, and the research notes of `sources`.
"""
import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decide as DC  # noqa: E402
import experiment as X  # noqa: E402
import keep3  # noqa: E402
import metrics  # noqa: E402
import plan as PL  # noqa: E402
import project as P  # noqa: E402
from run import COLS  # noqa: E402

T = tempfile.mkdtemp(prefix="ar_verdict_")
RUNS = ("V.a0", "V.a1", "V.c0", "V.c1")
CARDS = dict(a=dict(Q=0.76, F=0.9), b=dict(Q=0.74, F=0.9), ca=dict(Q=0.73, F=0.9), cb=dict(Q=0.72, F=0.9))   # effect a - c = +0.0250 on two agreeing seeds: KEEP
BASE = dict(id="E1", title="t", kind="variant", metric="Q", direction="higher", band=0.01, prediction=dict(value=0.03, vs="c"), control="c", rule=[dict(metric="Q", min_effect=0.01)],
            evaluate=dict(split="val"), pairing="position",
            arms=dict(a=dict(cards=["cards.json#a", "cards.json#b"], run_ids=["V.a0", "V.a1"]), c=dict(cards=["cards.json#ca", "cards.json#cb"], run_ids=["V.c0", "V.c1"])))
NOTE = "# {n} t\n\n- title: T {n}\n- relevance: 2\n- summary: It does a. It matters because b.\n- status: checked\n- paper: https://x.org/{n}\n- code: -\n- verdict: use\n- result: -\n"
ENV = dict(os.environ, AR_PROJECT=T)
for k in ("AR_LOCK_OWNER", "AR_LOCK_WAIT_S"):
    ENV.pop(k, None)
CARD_MD = f"{T}/experiments/E1_x/experiment.md"


def ledger(rows):
    with open(f"{T}/LEDGER.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])


def rows():
    return {r["id"]: r for r in csv.DictReader(open(f"{T}/LEDGER.csv", newline="", encoding="utf-8"))}


def put_card(exp, name="E1_x"):
    os.makedirs(f"{T}/experiments/{name}", exist_ok=True)
    path = f"{T}/experiments/{name}/experiment.md"
    open(path, "w", encoding="utf-8", newline="\n").write(f"# {exp['id']} t\n\n```json card\n{json.dumps(exp, indent=1)}\n```\n")
    return path


def record(rid, sha):
    os.makedirs(f"{T}/runs/{rid}", exist_ok=True)
    json.dump(dict(id=rid, status="done", **({"card_sha": sha} if sha else {})), open(f"{T}/runs/{rid}/run.json", "w"))


def fresh(exp=BASE, sha=None):
    """an empty problem: the cards, the four runs ok in the ledger, the card of E1; sha: the card_sha every run record holds (None: records without one)"""
    for d in ("experiments", "results", "literature", "runs"):
        shutil.rmtree(f"{T}/{d}", ignore_errors=True)
    for f in os.listdir(T):
        if f.endswith((".lock", ".csv", ".md")):
            os.remove(f"{T}/{f}")
    json.dump(dict(tag_prefix="v", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md",
                   experiment_dirs=["experiments", "results/experiments"]), open(f"{T}/project.json", "w"))
    json.dump(CARDS, open(f"{T}/cards.json", "w"))
    ledger([dict(id=r, status="ok") for r in RUNS])
    for r in RUNS:
        record(r, sha)
    return put_card(exp)


def decide(*args, wait=None):
    env = dict(ENV, **({"AR_LOCK_WAIT_S": str(wait)} if wait else {}))
    r = subprocess.run([sys.executable, f"{HERE}/decide.py", *args], capture_output=True, text=True, env=env, cwd=T)
    return r.returncode, r.stdout + r.stderr


def decision():
    return json.load(open(f"{T}/experiments/E1_x/decision.json", encoding="utf-8"))


def says(text, *words):
    for w in words:
        assert w in text, (w, text)


def raises(fn, *words):
    try:
        fn()
    except (SystemExit, ValueError) as e:
        says(str(e), *words)
        return
    raise AssertionError("must be refused")


def mutated(**kw):
    exp = json.loads(json.dumps(BASE))
    for k, v in kw.items():
        exp[k] = v
    return exp


# ------------------------------------------------------------------ 1. a rule fixed before the run


def t_rule_sha():
    e = X.rule_sha(BASE)
    assert re.fullmatch(r"[0-9a-f]{16}", e), e
    other = mutated(title="another title", card="H1.1", kill="loss@10 > 5", budget_gpu_h=9, seeds=[0, 1], sources=["a/b"])      # not part of the rule
    for arm in other["arms"].values():
        arm.update(recipe="r", base="b", settings={"lr": 1}, seeds=[0, 1], names=["x"], note="n")
    other["arms"]["a"]["run_ids"].append("V.a2")
    other["arms"]["a"]["cards"] = ["cards.json#b", "cards.json#a"]
    assert X.rule_sha(other) == e, "recipes, cards, names, run ids, seeds, base, settings, kill, budget and the title are not the rule"
    assert X.rule_sha(dict(reversed(list(BASE.items())))) == e, "the key order of the card is not the rule"
    changes = dict(band=0.02, metric="F", direction="lower", control="a", kind="block", deterministic=True, pairing="seed", variant="a", id="E2",
                   prediction=dict(value=0.04, vs="c"), rule=[dict(metric="Q", min_effect=0.02)], evaluate=dict(split="test"))
    for k, v in changes.items():
        assert X.rule_sha(mutated(**{k: v})) != e, f"{k} is part of the rule"
    assert X.rule_sha(mutated(arms=dict(BASE["arms"], d=dict(recipe="x")))) != e, "an arm letter is part of the rule"


def t_card_for_run():
    fresh()
    os.makedirs(f"{T}/results/experiments/E2_y")
    exp2 = mutated(id="E2", arms=dict(a=dict(recipe="r", run_ids=["V.z0"]), c=dict(recipe="r", run_ids="V.z0V.z1")))      # a run_ids that is no list holds no run
    open(f"{T}/results/experiments/E2_y/experiment.md", "w").write(f"# E2\n\n```json card\n{json.dumps(exp2)}\n```\n")
    os.makedirs(f"{T}/experiments/E3_bad")
    open(f"{T}/experiments/E3_bad/experiment.md", "w").write("# E3\n\n```json card\n{\"id\": }\n```\n")                                  # a broken card is skipped
    os.makedirs(f"{T}/experiments/E4_bin")
    open(f"{T}/experiments/E4_bin/experiment.json", "wb").write(b"\xff\xfe")                                                        # so is an unreadable one
    got = X.card_for_run("V.a1", T)
    assert got and got[0]["id"] == "E1" and got[1] == CARD_MD, got
    got = X.card_for_run("V.c1", T)
    assert got and got[0]["id"] == "E1", "any arm"
    got = X.card_for_run("V.z0", T)
    assert got and got[0]["id"] == "E2" and got[1].replace("\\", "/").endswith("results/experiments/E2_y/experiment.md"), "the second card folder is searched too"
    assert X.card_for_run("V.z1", T) is None and X.card_for_run("V.a", T) is None and X.card_for_run("nope", T) is None, "a whole id, in a list"
    assert X.card_for_run("V.a0", f"{T}/no/such/problem") is None
    old = os.environ["AR_PROJECT"]
    os.environ["AR_PROJECT"] = f"{T}/no/such/project"       # no problem folder at all: P.root() exits, card_for_run does not raise
    try:
        assert X.card_for_run("V.a0") is None
    finally:
        os.environ["AR_PROJECT"] = old
    assert X.card_for_run("V.a0")[0]["id"] == "E1", "root=None means the problem of AR_PROJECT"


def t_gate():
    sha = X.rule_sha(BASE)
    fresh(sha=sha)
    ledger([dict(id="V.a0", status="ok"), dict(id="V.a1", status="failed"), dict(id="V.c0", status="ok"), dict(id="V.c1", status="ok")])
    rc, out = decide("--experiment", CARD_MD, "--write")
    says(out, "run V.a1 has status failed: nothing to decide yet")
    assert rc == 1 and not os.path.exists(f"{T}/experiments/E1_x/decision.json") and not os.path.exists(f"{T}/experiments/E1_x/results.md"), "a refusal writes nothing"
    ledger([dict(id="V.a0", status="ok"), dict(id="V.c0", status="ok"), dict(id="V.c1", status="planned")])
    rc, out = decide("--experiment", CARD_MD)
    says(out, "run V.a1 has status missing: nothing to decide yet", "run V.c1 has status planned: nothing to decide yet")
    assert rc == 1, "the control arm is gated too; a missing row too"
    ledger([dict(id=r, status="ok") for r in RUNS])
    rc, out = decide("--experiment", CARD_MD, "--write")
    d = decision()
    assert rc == 0 and d["verdict"] == "KEEP" and d["rule_sha"] == sha and d["registered"] is True and d["by_hand"] == "" and "rule not registered" not in d["why"], (out, d)
    text = open(f"{T}/experiments/E1_x/results.md", encoding="utf-8").read()
    says(text, "**Verdict by the rule as written:** KEEP", f"`{sha}`")
    assert rows()["V.a0"]["decision"].startswith("decide.py ") and rows()["V.c0"]["decision"] == "", "the variant rows carry the verdict, the control rows do not"
    # the rule edited after the launch: refused, nothing rewritten; a change of what is not the rule (a recipe, the run ids) is fine
    before = open(f"{T}/experiments/E1_x/decision.json", "rb").read()
    put_card(mutated(band=0.02))
    rc, out = decide("--experiment", CARD_MD, "--write")
    says(out, "the rule changed after run V.a0 was launched")
    assert rc == 1 and open(f"{T}/experiments/E1_x/decision.json", "rb").read() == before
    put_card(dict(BASE, arms={k: dict(v, recipe="new words") for k, v in BASE["arms"].items()}, kill="x", title="y"))
    assert decide("--experiment", CARD_MD)[0] == 0, "recipe, kill and title are not the rule"
    # --by-hand: the rule changed, and a run that is not ok: decided, labelled in decision.json, the ledger token and results.md
    put_card(mutated(band=0.02))
    ledger([dict(id="V.a0", status="ok", decision="my note"), dict(id="V.a1", status="failed"), dict(id="V.c0", status="ok"), dict(id="V.c1", status="ok")])
    rc, out = decide("--experiment", CARD_MD, "--write", "--by-hand", "the band was a typo | fixed")
    d = decision()
    assert rc == 0 and d["by_hand"] == "the band was a typo | fixed" and d["registered"] is False, (out, d)
    says(open(f"{T}/experiments/E1_x/results.md", encoding="utf-8").read(), "**Verdict (by hand: the band was a typo | fixed):** KEEP")
    cell = rows()["V.a0"]["decision"]
    assert cell.startswith("decide.py ") and ", by hand: the band was a typo / fixed) | my note" in cell and DC.verdict_of(cell) == "KEEP", cell     # a '|' of the reason would break the cell
    assert decide("--experiment", CARD_MD, "--by-hand", "  ")[0] == 2 and decide("--variant", "1", "--control", "0", "--by-hand", "x")[0] == 2, "a reason is needed; typed scores are by hand anyway"
    # old records without a card_sha: decided, and said so
    fresh(sha=None)
    rc, out = decide("--experiment", CARD_MD, "--write")
    d = decision()
    assert rc == 0 and d["registered"] is False and d["why"].endswith("rule not registered at launch (no card_sha in the run records)"), (out, d["why"])
    says(open(f"{T}/experiments/E1_x/results.md", encoding="utf-8").read(), "**Verdict (rule not registered at launch):** KEEP")
    # one record with another rule is enough, also the control's
    fresh(sha=None)
    record("V.c1", "0123456789abcdef")
    rc, out = decide("--experiment", CARD_MD)
    says(out, "the rule changed after run V.c1 was launched")
    assert rc == 1
    # decide_experiment itself has no gate (X.check calls it); the gate is gated_decision
    raises(lambda: DC.gated_decision(BASE, T, [dict(id="V.a0", status="ok")], f"{T}/runs"), "run V.a1 has status missing")


def t_previous():
    fresh(sha=X.rule_sha(BASE))
    assert decide("--experiment", CARD_MD, "--write")[0] == 0 and decision()["previous"] == []
    first = decision()
    json.dump(CARDS | dict(a=dict(Q=0.80, F=0.9)), open(f"{T}/cards.json", "w"))
    assert decide("--experiment", CARD_MD, "--write")[0] == 0
    prev = decision()["previous"]
    assert len(prev) == 1 and prev[0] == {k: first[k] for k in ("verdict", "effect", "date", "rule_sha", "by_hand")} and decision()["effect"] != first["effect"], prev
    out = f"{T}/experiments/E1_x/decision.json"
    json.dump(dict(verdict="CUT", effect=-1, date="d", rule_sha="r", by_hand="", previous=[dict(verdict="KEEP", effect=i) for i in range(10)]), open(out, "w"))
    got = DC.previous_verdicts(out)
    assert len(got) == 10 and got[-1]["verdict"] == "CUT" and got[0]["effect"] == 1, "the cap is 10, the oldest goes"
    open(out, "wb").write(b"\xff\xfe not json")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert DC.previous_verdicts(out) == []
    says(buf.getvalue(), "warning", "cannot be read")


# ------------------------------------------------------------------ 2. the decision cell


def t_decision_cell():
    tok = DC.decision_token("KEEP", "2026-11-10", 0.0123)
    assert tok == "decide.py 2026-11-10: KEEP (effect +0.0123)" and DC.TOKEN.fullmatch(tok)
    prose = "the last attempt was a CUT; " + "x" * 600
    cell = DC.ledger_decision(prose, tok)
    assert cell.startswith(tok + " | the last attempt was a CUT; ") and len(cell) == DC.DECISION_CAP and DC.verdict_of(cell) == "KEEP", (len(cell), cell[:120])
    assert DC.split_decision(cell)[1] == tok and DC.split_decision(cell)[0].startswith("the last attempt was a CUT;")
    longtok = DC.decision_token("KEEP", "2026-11-10", 0.0123, ["by hand: " + "y" * 500])
    assert "y" * 121 not in longtok and DC.ledger_decision(prose, longtok).startswith(longtok) and len(DC.ledger_decision(prose, longtok)) <= DC.DECISION_CAP, "a note of the token is capped, the token is never cut"
    again = DC.ledger_decision(cell, DC.decision_token("CUT", "2026-11-11", -0.02))
    assert again.count("decide.py") == 1 and again.startswith("decide.py 2026-11-11: CUT (effect -0.0200) | the last attempt") and DC.verdict_of(again) == "CUT", "an earlier token is replaced, the prose kept"
    old = DC.ledger_decision("my note | decide.py 2026-11-01: TIE (effect +0.00)", tok)
    assert old == tok + " | my note", old
    assert DC.ledger_decision("decide.py: TIE first, then KEEP (effect +0.02)", tok) == tok, "a verdict typed by hand without the dated token is replaced, not kept as prose"
    assert DC.verdict_of("KEEP then CUT") == "CUT" and DC.verdict_of("decide.py: TIE first, then KEEP") == "KEEP" and DC.verdict_of("") == "", "no token: the last verdict word"
    # every reader, both orders
    first = f"{tok} | {prose}"
    behind = f"{prose[:80]} | {tok}"
    for c in (tok, first, behind, DC.ledger_decision(prose, tok)):
        assert DC.verdict_of(c) == "KEEP", c
        assert PL.status(None, [dict(item="E5.a", node="E5", state="done")], [dict(id="E5.a", status="ok", decision=c, started="1")])[0]["verdict"] == "KEEP", c
    fresh()
    ledger([dict(id="E5.a", status="ok", hypothesis="cand beats it", decision=f"{tok} | the earlier attempt was a CUT, read it"),         # the old rule took the last verdict word: CUT
            dict(id="E6.a", status="ok", hypothesis="cand beats it", decision=f"an earlier TIE | {tok}"),
            dict(id="E7.a", status="ok", hypothesis="cand beats it", decision="decide.py 2026-11-10: CUT (effect -0.0200) | KEEP is what I hoped")])
    cand = dict(name="cand", runs=["rcand"])
    ns = lambda ledger_id: type("A", (), dict(by_hand="", ledger=ledger_id))()  # noqa: E731
    assert keep3.evidence(ns("E5"), cand) == "ledger E5 KEEP" and keep3.evidence(ns("E6"), cand) == "ledger E6 KEEP"
    raises(lambda: keep3.evidence(ns("E7"), cand), "holds the verdict KEEP")
    # metrics counts a decide.py row through its decision file only, and reads the prose without the token (the token's by-hand reason is no part of the prediction)
    ledger([dict(id="E1.a", status="ok", decision="rule met; prediction met | decide.py 2026-11-10: KEEP (effect +0.0200)"),
            dict(id="E2.a", status="ok", decision="decide.py 2026-11-10: KEEP (effect +0.0200, by hand: prediction missed here) | rule met; prediction met"),
            dict(id="E3.a", status="ok", decision="rule not met; prediction missed")])
    cfg = dict(P.DEFAULTS, ledger="LEDGER.csv", loops={"1": [1, 9]})
    L = metrics.collect(T, cfg)["1"]
    assert (L["pred_met"], L["pred_n"], L["KEEP"], L["CUT"], L["other"]) == (2, 3, 0, 1, 0), L


# ------------------------------------------------------------------ 3. the checks of a card


def t_check():
    fresh()
    assert X.check(BASE, CARD_MD) == [], X.check(BASE, CARD_MD)
    clause = lambda **kw: mutated(rule=[dict(metric="Q", min_effect=0.01, **kw)])  # noqa: E731
    found = lambda exp: " | ".join(X.check(exp, CARD_MD))  # noqa: E731
    for bad in (0, -1, 3, True, 1.5, "2"):
        says(found(clause(subsets=["n", "b"], at_least=bad)), "at_least must be a whole number from 1 to 2")
    says(found(clause(at_least=2)), "from 1 to 1")
    assert "at_least" not in found(clause(at_least=1)) and "at_least" not in found(clause(subsets=["n", "b"], at_least=2)) and "at_least" not in found(clause(subsets=["n", "b"], at_least=1))
    says(found(clause(subsets=["n", "n"])), "subsets must be unique")
    says(found(clause(subsets="n")), "subsets must be a list")
    says(found(clause(subsets=[1, 2])), "subsets must be a list")
    says(found(mutated(rule=[dict(metric="Q", min_effect=-0.01)])), "min_effect must be >= 0")
    says(found(mutated(rule=[dict(metric="Q", min_effect=0.01), dict(metric="F", max_loss=-1)])), "max_loss must be >= 0")
    says(found(mutated(rule=[dict(metric="Q", min_effect=float("nan"))])), "finite")
    assert "must be >= 0" not in found(mutated(rule=[dict(metric="Q", min_effect=0), dict(metric="F", max_loss=0)]))
    # a card file listed twice counts as two seeds
    twice = mutated(arms=dict(BASE["arms"], a=dict(BASE["arms"]["a"], cards=["cards.json#a", "experiments/../cards.json#a"])))
    says(found(twice), "arm a: card experiments/../cards.json#a is listed twice")
    across = mutated(arms=dict(BASE["arms"], c=dict(BASE["arms"]["c"], cards=["cards.json#a", "cards.json#cb"])))
    says(found(across), "card cards.json#a is listed in arm a and in arm c")
    raises(lambda: DC.decide_experiment(twice, T), "listed twice")
    raises(lambda: DC.decide_experiment(across, T), "listed in arm a and in arm c")
    put_card(twice)
    rc, out = decide("--experiment", CARD_MD)
    says(out, "experiment file not ready", "is listed twice")
    assert rc == 1, "decide.py refuses it as check does"
    # run_ids must be a list of strings (the gate reads them); sources: <topic slug>/<note id> (the baseline above lists one file twice with other inner keys: that is fine)
    says(found(mutated(arms=dict(BASE["arms"], a=dict(BASE["arms"]["a"], run_ids="V.a0")))), "run_ids must be a list")
    assert X.check(mutated(sources=["lanes/n1", "a.b/c_d-e"]), CARD_MD) == [] and X.check(mutated(sources=[]), CARD_MD) == []
    for bad in (["a"], ["a/b/c"], ["../x/y"], ["a/.."], ["a b/c"], [5], "a/b", ["a/"], ["/b"], [" a/b"], ["a/b\n"], [["a/b"]]):
        says(found(mutated(sources=bad)), "sources must be")


# ------------------------------------------------------------------ 4. --write: order, atomic files, warnings instead of an abort


def t_write_failures():
    sha = X.rule_sha(BASE)
    fresh(sha=sha)
    assert decide("--experiment", CARD_MD, "--write")[0] == 0
    dpath, rpath = f"{T}/experiments/E1_x/decision.json", f"{T}/experiments/E1_x/results.md"
    before = (open(dpath, "rb").read(), open(rpath, "rb").read(), rows()["V.a0"]["decision"])
    # the ledger comes first: a ledger that is locked stops the write before decision.json and results.md are touched
    json.dump(CARDS | dict(a=dict(Q=0.80, F=0.9)), open(f"{T}/cards.json", "w"))
    open(f"{T}/LEDGER.csv.lock", "w").write(f"{os.getpid()} {time.time()} held")
    rc, out = decide("--experiment", CARD_MD, "--write", wait=0.5)
    os.remove(f"{T}/LEDGER.csv.lock")
    says(out, "the ledger note of V.a0 failed")
    assert rc == 1 and (open(dpath, "rb").read(), open(rpath, "rb").read(), rows()["V.a0"]["decision"]) == before, "ledger first: nothing else written when it fails"
    # a results.md that is not UTF-8, a schedule that is locked: warnings, the verdict is written, exit 0
    open(rpath, "wb").write(b"# E1 result - KEEP\n\xff\xfe broken\n")
    import schedule as SC
    SC.save([dict(item="E1.train", node="E1", kind="train", state="done")], f"{T}/schedule.csv")
    open(f"{T}/schedule.csv.lock", "w").write(f"{os.getpid()} {time.time()} held")
    rc, out = decide("--experiment", CARD_MD, "--write", wait=0.5)
    os.remove(f"{T}/schedule.csv.lock")
    says(out, "warning: results.md not written", "warning: schedule link not written")
    assert rc == 0 and decision()["effect"] == 0.045 and DC.TOKEN.search(rows()["V.a0"]["decision"]) and open(rpath, "rb").read().startswith(b"# E1 result - KEEP\n\xff\xfe"), out
    # decision.json and results.md go through atomic_write (temporary file and replace), and leave no temporary file
    os.remove(rpath)
    seen, real = [], P.atomic_write
    P.atomic_write = lambda path, text: (seen.append(os.path.basename(path)), real(path, text))[1]
    try:
        d = DC.gated_decision(BASE, T, list(rows().values()), f"{T}/runs")
        with contextlib.redirect_stdout(io.StringIO()):
            assert DC.write_records(CARD_MD, BASE, d, T) == 0
    finally:
        P.atomic_write = real
    assert "decision.json" in seen and "results.md" in seen, seen
    assert not [f for f in os.listdir(f"{T}/experiments/E1_x") if ".tmp." in f], os.listdir(f"{T}/experiments/E1_x")


# ------------------------------------------------------------------ 5. the typed form


def t_typed():
    fresh()
    ledger([dict(id="E7.t", status="ok", decision="my note"), dict(id="E8.t", status="failed", decision="old")])
    args = ("--variant", "0.76,0.75", "--control", "0.73,0.72", "--write", "--id")
    rc, out = decide(*args, "E7.t")
    want = r"decide\.py \d{4}-\d\d-\d\d: KEEP \(effect \+0\.0300, typed scores, band 0\.010, by hand: typed scores\) \| my note"
    assert rc == 0 and re.fullmatch(want, rows()["E7.t"]["decision"]), (out, rows()["E7.t"]["decision"])
    assert decide(*args, "E7.t")[0] == 0 and rows()["E7.t"]["decision"].count("decide.py") == 1, "a second write replaces the token"
    cell = rows()["E7.t"]["decision"]
    assert PL.status(None, [dict(item="E7.t", node="E7.t", state="done")], [dict(id="E7.t", status="ok", decision=cell, started="1")])[0]["verdict"] == "KEEP", "plan.py reads the typed token"
    ledger([dict(id="E7.t", status="ok", decision=cell)])
    assert metrics.collect(T, dict(P.DEFAULTS, ledger="LEDGER.csv", loops={"1": [1, 9]}))["1"]["other"] == 0, "metrics.py does not count the typed token as 'other'"
    for rid, word in (("E8.t", "has status failed: nothing to decide yet"), ("E9.t", "no ledger row E9.t")):
        ledger([dict(id="E8.t", status="failed", decision="old")])
        rc, out = decide(*args, rid)
        says(out, word)
        assert rc == 1 and rows()["E8.t"]["decision"] == "old" and "E9.t" not in rows(), (rc, out)


# ------------------------------------------------------------------ 6. sources: the research notes of the card


def t_sources():
    exp = mutated(sources=["lanes/n1", "lanes/n2", "nowhere/x", "lanes/missing", "lanes/n1"])
    fresh(exp, sha=X.rule_sha(exp))
    topic = f"{T}/literature/topics/lanes"
    for n in ("n1", "n2"):
        os.makedirs(f"{topic}/{n}")
        open(f"{topic}/{n}/note.md", "w").write(NOTE.format(n=n))
    rc, out = decide("--experiment", CARD_MD, "--write")
    says(out, "warning: topic folder", "nowhere", "warning: lanes/missing", "2 notes -> ")
    assert rc == 0
    for n in ("n1", "n2"):
        assert "- result: E1: KEEP effect +0.0250\n" in open(f"{topic}/{n}/note.md").read(), n          # the text comes from the decision: id, verdict, effect
    assert "E1: KEEP effect +0.0250" in open(f"{topic}/table.md").read(), "the table is made again"
    # a note with a broken header: a warning, the other notes are updated, the table is not made (topic_table removes a stale one), exit 0
    os.makedirs(f"{topic}/n3")
    open(f"{topic}/n3/note.md", "w").write("# n3\n- title: only\n")
    exp2 = mutated(sources=["lanes/n3", "lanes/n1"])
    put_card(exp2)
    for r in RUNS:
        record(r, X.rule_sha(exp2))
    rc, out = decide("--experiment", CARD_MD, "--write")
    says(out, "warning: lanes/n3:", "warning: table of lanes not made")
    assert rc == 0 and not os.path.exists(f"{topic}/table.md"), out


SECTIONS = [t_rule_sha, t_card_for_run, t_gate, t_previous, t_decision_cell, t_check, t_write_failures, t_typed, t_sources]


def main():
    failed = []
    old = os.environ.get("AR_PROJECT")
    os.environ["AR_PROJECT"] = T
    try:
        for fn in SECTIONS:
            try:
                fn()
                print(f"ok    {fn.__name__}")
            except (Exception, SystemExit):  # noqa: BLE001  (every section runs, also after a refusal that exits; the failures are listed at the end)
                failed.append(fn.__name__)
                print(f"FAIL  {fn.__name__}\n" + "".join(traceback.format_exc().splitlines(True)[-6:]))
    finally:
        if old is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old
        shutil.rmtree(T, ignore_errors=True)
    if failed:
        print("verdict selftest FAILED: " + ", ".join(failed))
        return 1
    print("verdict selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
