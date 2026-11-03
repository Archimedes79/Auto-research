"""Gate of the result-report builder (results/build/): a tiny report from a fixture of known numbers, in a temporary folder (CPU, no GPU, no data, a few seconds).
  A  every decimal number of the page and of the mail is in a source file (autoresearch/tools/check_numbers.py --strict): the fixture's tables, train table, compare output, timing and
     config, plus the differences that the page computes, written here by independent decimal arithmetic
  B  every cell is where it belongs: the bundle table, the weak-spot table and the summary table against hand-written rows (new / control columns, difference, paired bootstrap, the
     reading in both directions, a head the control lacks, the ms row, the weak order, the flag, the train - validation gap), the chapter markers, the mail table and cost lines
  C  a missing input stops the build with a message and writes no page (missing evaluation, compare output missing / of other runs / disagreeing with the tables / without a group,
     missing key, missing or non-idle timing, unknown fact in a text, unknown section)
  D  mutants: the builder is copied, one rule is broken, and the checks above must notice (each of the listed ones must be killed)
  python tests/test_report_build.py          the gate
  python tests/test_report_build.py --real   by hand (needs the real files): the v6 config rebuilds results/t16/report_big.html, the mail and the tables (see results/build/README.md)"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "..", "results", "build")
CHECK = os.path.join(HERE, "..", "..", "autoresearch", "tools", "check_numbers.py")
sys.path.insert(0, BUILD)

LANE05, LANE1 = "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1"
CL1, CLA = "seg_fov/seg/dt+0.00/centreline/all/all/iou@1", "seg_fov/seg/dt+0.00/centreline:dir/all/all/angle_mae"
DRIV, WALK, PARK = (f"seg_fov/seg/dt+0.00/{x}/all/all/iou@0" for x in ("drivable_area", "walkway", "carpark_area"))
VEH, VEL = "detf/dt+0.00/vehicle/all/ap@2", "vel/dt+0.00/centre/field/moving/epe"
CTL = {LANE05: "0.4300", LANE1: "0.5500", DRIV: "0.7800", WALK: "0.4000", PARK: "0.3000", VEH: "0.8000", VEL: "1.1000"}
NEW = {LANE05: "0.4900", LANE1: "0.6100", CL1: "0.4000", CLA: "0.3000", DRIV: "0.7750", WALK: "0.3800", PARK: "0.3300", VEH: "0.8100", VEL: "1.0000"}
TRAIN = {LANE05: "0.6000", DRIV: "0.8000", WALK: "0.4500", PARK: "0.5000"}                     # the new run's score on its own training scenes
MS = {"ctl": 96.4, "new": 99.7}
COMPARE = """paired scene bootstrap (1000 resamples), b - a; a = runs/ctl, b = runs/new; val; 1000 common frames
seg.fov/dt+0.00/lane_divider/all/all/iou@0.5    a 0.4300  b 0.4900  b-a +0.0600  95% [+0.0300, +0.0900]  (excludes 0)  draws 1000/1000
seg.fov/dt+0.00/drivable_area/all/all/iou@0     a 0.7800  b 0.7750  b-a -0.0050  95% [-0.0200, +0.0100]  draws 1000/1000
seg.fov/dt+0.00/walkway/all/all/iou@0           a 0.4000  b 0.3800  b-a -0.0200  95% [-0.0300, -0.0100]  (excludes 0)  draws 1000/1000
detf.dt+0.00/vehicle/all/ap@2                   a 0.8000  b 0.8100  b-a +0.0100  95% [-0.0100, +0.0300]  draws 1000/1000
vel.dt+0.00/centre/field/moving/epe             a 1.1000  b 1.0000  b-a -0.1000  95% [-0.1500, -0.0500]  (excludes 0)  draws 1000/1000
"""
ROWS = [dict(name="lane divider IoU@0.5", key=LANE05, weak=True), dict(name="lane divider IoU@1", key=LANE1, weak=True),
        dict(name="centre-line IoU@1", key=CL1, weak=True, new_only=True), dict(name="centre-line angle error (rad)", key=CLA, lower=True, new_only=True),
        dict(name="drivable area IoU", key=DRIV, weak=True), dict(name="walkway IoU", key=WALK, weak=True), dict(name="car park IoU", key=PARK, weak=True),
        dict(name="vehicle AP@2 m", key=VEH, weak=True), dict(name="velocity EPE (m/s)", key=VEL, lower=True), dict(name="inference (ms per frame)", timing=True, lower=True)]
NOTEST, NOCTL = "n/a (no paired test for this metric)", "n/a (the control has no such value)"          # the paired cell of a row without a bootstrap: the difference column keeps the unpaired one
# the rows the bundle table must show: name, new, control, difference, paired b - a [interval], reading
EXPECT_TABLE = [["lane divider IoU@0.5", "0.490", "0.430", "+0.060", "+0.0600 [+0.0300, +0.0900]", "better"],
                ["lane divider IoU@1", "0.610", "0.550", "+0.060", NOTEST, "n/a"],
                ["centre-line IoU@1", "0.400", "-", "-", NOCTL, "n/a"],
                ["centre-line angle error (rad), lower is better", "0.300", "-", "-", NOCTL, "n/a"],
                ["drivable area IoU", "0.775", "0.780", "-0.005", "-0.0050 [-0.0200, +0.0100]", "not separated"],
                ["walkway IoU", "0.380", "0.400", "-0.020", "-0.0200 [-0.0300, -0.0100]", "worse"],
                ["car park IoU", "0.330", "0.300", "+0.030", NOTEST, "n/a"],
                ["vehicle AP@2 m", "0.810", "0.800", "+0.010", "+0.0100 [-0.0100, +0.0300]", "not separated"],
                ["velocity EPE (m/s), lower is better", "1.000", "1.100", "-0.100", "-0.1000 [-0.1500, -0.0500]", "better"],
                ["inference (ms per frame), lower is better", "99.7", "96.4", "+3.3", NOTEST, "n/a"]]
# the weak-spot table: output, new, control, train - validation gap of the new run, paired, flag (lowest of the new run first; flagged below weak_below = 0.5)
EXPECT_WEAK = [["car park IoU", "0.330", "0.300", "+0.170", NOTEST, "weak"], ["walkway IoU", "0.380", "0.400", "+0.070", "-0.0200 [-0.0300, -0.0100]", "weak"],
               ["centre-line IoU@1", "0.400", "-", "-", NOCTL, "weak"], ["lane divider IoU@0.5", "0.490", "0.430", "+0.110", "+0.0600 [+0.0300, +0.0900]", "weak"],
               ["lane divider IoU@1", "0.610", "0.550", "-", NOTEST, ""], ["drivable area IoU", "0.775", "0.780", "+0.025", "-0.0050 [-0.0200, +0.0100]", ""],
               ["vehicle AP@2 m", "0.810", "0.800", "-", "+0.0100 [-0.0100, +0.0300]", ""]]
EXPECT_SUMMARY = [[r[0], r[1], r[2], r[5]] for r in EXPECT_TABLE]
BANK = """@@@ lead
Test report. State @@head@@, @@frames@@ frames.

@@@ summary
<div class=card>Summary of the test.</div>
<h3>Key numbers @@new_label@@ against @@ctl_label@@</h3>
@@key_table@@

@@@ results_intro
<p>Results intro.</p>

@@@ results_cov_note
<p class=small>Coverage note.</p>

@@@ results_gen_note
<p>Generalisation note.</p>
"""
CHAPTER = "Intro of the chapter.\n\n@@TABLE@@\n\n- first point\n- second point\n\n@@WEAK@@\n\n| a | b |\n|---|---|\n| x | y |\n"
MAIL = "Hi\n@@MAIL_TABLE@@\ncost\n@@MAIL_COST@@\nstate @@head@@\n"


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8", newline="\n").write(text)


def edit(root, rel, old, new):
    p = os.path.join(root, rel)
    s = open(p, encoding="utf-8").read()
    assert old in s, (rel, old)
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))


def timing(total, idle=True):
    return ('{"idle": %s, "inference_ms": {"total_ms": %s, "unattributed_ms": 1.0, "per_frame_ms": {"modA": %s, "modB": %s}, "calls": {"modA": 1, "modB": 1}}}'
            % ("true" if idle else "false", total, round(total - 40.0, 1), 40.0))


def fixture(root):
    """a problem folder of two runs with known numbers, the compare output, a config, texts"""
    lines = lambda d: "".join(f"{k}   {v}\n" for k, v in d.items())          # noqa: E731
    write(root, "runs/ctl/eval/by_n/table_n1000.txt", lines(CTL))
    write(root, "runs/new/eval/by_n/table_n1000.txt", lines(NEW))
    write(root, "runs/new/eval/table_train.txt", lines(TRAIN))
    for r in ("ctl", "new"):
        write(root, f"runs/{r}/eval/timing_modules.json", timing(MS[r]))
    write(root, "cmp.txt", COMPARE)
    write(root, "chapters/text.txt", BANK)
    write(root, "chapters/b.md", CHAPTER)
    write(root, "chapters/mail.txt", MAIL)
    write(root, "report_runs.json", json.dumps(config(), indent=1))
    write(root, "labels.txt", "static numbers of the table labels: dt -0.5 s, dt +0.5 s\n")


def config():
    return {"title": "Test report", "out": "out", "frames": 1000, "text": ["chapters/text.txt"], "sections": ["summary", "bundle", "results"],
            "runs": [{"id": "ctl", "label": "control run", "role": "baseline"}, {"id": "new", "label": "new run", "role": "arm"}],
            "bundle": {"title": "The bundle", "run": "new", "control": "ctl", "new_label": "t.new", "control_label": "t.ctl", "compare": "cmp.txt", "chapter": "chapters/b.md",
                       "band": 0.010, "weak_below": 0.5, "rows": ROWS},
            "summary": {"table": "bundle"}, "mail": {"narrative": "chapters/mail.txt"}}


def build(build_dir, root, out="out"):
    """-> (exit code, stderr); the page is root/out/report.html"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, os.path.join(build_dir, "build_report.py"), "--config", os.path.join(root, "report_runs.json"), "--root", root, "--out", os.path.join(root, out),
                        "--no-check"], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def cells(table_html):
    return [[html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)] for row in re.findall(r"<tr>(.*?)</tr>", table_html, re.S)]


def section(page, sec_id):
    return page.split(f"<h2 id='{sec_id}'>")[1].split("<h2 id=")[0]


def derived(root):
    """the differences the page computes, by independent decimal arithmetic: new - control per key, train - validation per key, the ms difference"""
    dec, out = lambda s: Decimal(str(s)), []                                          # noqa: E731
    q = lambda x: f"{x.quantize(Decimal('0.001')):+}"                                 # noqa: E731
    out += [q(dec(NEW[k]) - dec(CTL[k])) for k in CTL if k in NEW]
    out += [q(dec(TRAIN[k]) - dec(NEW[k])) for k in TRAIN]
    out.append(f"{dec(MS['new']) - dec(MS['ctl']):+.1f}")
    write(root, "derived.txt", "\n".join(out) + "\n")


def failures(root, out="out"):
    """every way the output of a build can be wrong: [] when it is right"""
    bad, page = [], open(os.path.join(root, out, "report.html"), encoding="utf-8").read()
    derived(root)
    srcs = [os.path.join(root, p) for p in ("runs/ctl/eval/by_n/table_n1000.txt", "runs/new/eval/by_n/table_n1000.txt", "runs/new/eval/table_train.txt", "cmp.txt",
                                            "runs/ctl/eval/timing_modules.json", "runs/new/eval/timing_modules.json", "report_runs.json", "derived.txt", "labels.txt")]
    for doc in ("report.html", "mail.txt"):
        r = subprocess.run([sys.executable, CHECK, os.path.join(root, out, doc)] + srcs + ["--strict"], capture_output=True, text=True)
        if r.returncode:
            bad.append("A: numbers of " + doc + " not in the sources: " + " | ".join(r.stdout.splitlines()[:4]))
    bundle, summary = section(page, "bundle"), section(page, "s1")
    tabs = re.findall(r"<table.*?</table>", bundle, re.S)
    got = [cells(t) for t in tabs]
    if len(got) < 3:
        return bad + [f"B: the bundle chapter has {len(got)} tables, expected 3 (numbers, weak spots, the chapter's own)"]
    if got[0][1:] != EXPECT_TABLE:
        bad.append(f"B: bundle table {got[0][1:]}")
    if got[0][0] != ["metric", "t.new", "t.ctl", "difference", "paired b - a [95 % interval]", "reading"]:
        bad.append(f"B: bundle table header {got[0][0]}")
    if got[1][1:] != EXPECT_WEAK:
        bad.append(f"B: weak-spot table {got[1][1:]}")
    if cells(re.findall(r"<table.*?</table>", summary, re.S)[0])[1:] != EXPECT_SUMMARY:
        bad.append("B: summary table")
    if not (bundle.index("Intro of the chapter") < bundle.index("The numbers (VAL") < bundle.index("<ul><li>first point</li><li>second point</li></ul>") < bundle.index("Weak spots")
            < bundle.index("<th class=l>a</th>")):
        bad.append("B: the chapter's text and the two tables are not in the order of the markers")
    if '<span class="delta good">+0.060</span>' not in bundle or '<span class="delta bad">-0.020</span>' not in bundle or '<span class="delta">-0.005</span>' not in bundle:
        bad.append("B: the difference is not coloured by the direction and the band")
    mail = open(os.path.join(root, out, "mail.txt"), encoding="utf-8").read()
    want = [("lane divider IoU@0.5", ["0.430", "0.490"]), ("centre-line IoU@1", ["-", "0.400"]), ("walkway IoU", ["0.400", "0.380"])]
    for name, vals in want:
        line = next((ln for ln in mail.splitlines() if ln.startswith(name)), "")
        if line.split()[-2:] != vals:
            bad.append(f"B: mail row '{name}': '{line}'")
    for label, ms, a in (("control run", "96.4", "56.4"), ("new run", "99.7", "59.7")):
        if not re.search(rf"^ +{label} +{re.escape(ms)} ms per frame [(]idle GPU[)]; largest modules: modA {re.escape(a)}, modB 40.0$", mail, flags=re.M):
            bad.append(f"B: mail cost line of {label}")
    return bad


def safe_failures(root):
    try:
        return failures(root)
    except Exception as e:                                         # a page that the checks cannot even read is a failure too
        return [f"crash while checking: {e!r}"]


# ---- C: a missing input stops the build (scenario: what to break, a word of the message)
def drop(path):
    return lambda r: os.remove(os.path.join(r, path))


def sub(path, old, new):
    return lambda r: edit(r, path, old, new)


def cut_group(r):
    p = os.path.join(r, "cmp.txt")
    kept = "".join(ln for ln in open(p, encoding="utf-8") if not ln.startswith("vel."))
    open(p, "w", encoding="utf-8").write(kept)


SCENARIOS = {"no eval of a run": (lambda r: shutil.rmtree(os.path.join(r, "runs/new/eval")), "no evaluation table"),
             "no chapter file": (drop("chapters/b.md"), "bundle chapter"),
             "no compare file": (drop("cmp.txt"), "bev_suite.py compare"),
             "compare of other runs": (sub("cmp.txt", "b = runs/new", "b = runs/other"), "compare file is about"),
             "key missing in a run": (sub("runs/new/eval/by_n/table_n1000.txt", "walkway", "walkwax"), "has no key"),
             "compare disagrees with the table": (sub("cmp.txt", "b 0.4900", "b 0.4800"), "compare file says"),
             "compare lacks a group": (cut_group, "group 'vel'"),
             "no timing": (drop("runs/new/eval/timing_modules.json"), "timing_modules.json"),
             "timing not idle": (sub("runs/new/eval/timing_modules.json", '"idle": true', '"idle": false'), "idle GPU"),
             "unknown fact in a text": (sub("chapters/text.txt", "Summary of the test.", "Summary @@nonsense@@."), "nonsense"),
             "text block missing": (sub("chapters/text.txt", "@@@ results_intro", "@@@ results_intr0"), "results_intro"),
             "unknown section": (sub("report_runs.json", '"results"', '"nothere"'), "unknown section")}


def scenario_ok(build_dir, name, tmp):
    """True when the broken fixture stops the build with the expected message and writes no page"""
    root = os.path.join(tmp, "sc_" + name.replace(" ", "_"))
    fixture(root)
    SCENARIOS[name][0](root)
    rc, msg = build(build_dir, root)
    return rc != 0 and "report build FAILED" in msg and SCENARIOS[name][1] in msg and not os.path.exists(os.path.join(root, "out", "report.html")), msg


# ---- D: mutants: (name, file, old, new, how it must be noticed: "out" = the checks on a good fixture, or a scenario name)
MUTANTS = [("columns swapped", "bundle_numbers.py", 'vals = [side["table"].get(s["key"]) for side in (new, ctl)]', 'vals = [side["table"].get(s["key"]) for side in (ctl, new)]', "out"),
           ("difference reversed", "bundle_numbers.py", "diff = None if vals[0] is None or vals[1] is None else vals[0] - vals[1]", "diff = None if vals[0] is None or vals[1] is None else vals[1] - vals[0]", "out"),
           ("reading ignores the direction", "bundle_numbers.py", 'return "better" if (float(paired[2]) < 0) == row_lower else "worse"', 'return "better" if float(paired[2]) > 0 else "worse"', "out"),
           ("reading ignores the interval", "bundle_numbers.py", 'if not paired[5]:\n        return "not separated"', 'if False:\n        return "not separated"', "out"),
           ("compare key not mapped", "report_lib.py", 'return f"{m.group(1)}.{m.group(2)}/{m.group(3)}" if m else', 'return table_key if False else', "out"),
           ("weak order reversed", "bundle_numbers.py", "key=lambda r: r[\"new\"])", "key=lambda r: -r[\"new\"])", "out"),
           ("weak flag reversed", "bundle_numbers.py", '"weak" if r["new"] < threshold else ""', '"weak" if r["new"] > threshold else ""', "out"),
           ("weak lists every row", "bundle_numbers.py", 'if r["weak"] and r["new"] is not None', 'if r["new"] is not None', "out"),
           ("gap reversed", "build_report.py", "gaps[r[\"key\"]] = tr[r[\"key\"]] - val[r[\"key\"]]", "gaps[r[\"key\"]] = val[r[\"key\"]] - tr[r[\"key\"]]", "out"),
           ("ms with another precision", "bundle_numbers.py", 'f"{v:.1f}" if row["timing"]', 'f"{v:.2f}" if row["timing"]', "out"),
           ("chapter bullets not rendered", "build_report.py", "render_md(p, bullets=True)", "render_md(p, bullets=False)", "out"),
           ("a head of the new run only is refused", "bundle_numbers.py", 'not (s.get("new_only") and role == "control")', "True", "out"),
           ("missing input does not stop", "report_lib.py", "    if not os.path.isfile(path):\n        fail(", "    if False:\n        fail(", "no chapter file"),
           ("missing run is skipped", "build_report.py", 'if r.get("optional"):', "if True:", "no eval of a run"),
           ("compare of other runs accepted", "bundle_numbers.py", "if ids != (ctl[\"id\"], new[\"id\"]):", "if False:", "compare of other runs"),
           ("compare disagreeing with the table accepted", "bundle_numbers.py", "abs(v - float(p)) > 1e-3", "abs(v - float(p)) > 1e9", "compare disagrees with the table"),
           ("a missing group accepted", "bundle_numbers.py", 'ck.split(".")[0] not in groups', "False", "compare lacks a group"),
           ("a missing key accepted", "bundle_numbers.py", "if v is None and not test and not", "if False and not", "key missing in a run"),
           ("non-idle timing accepted", "bundle_numbers.py", 'if side["ms"] is None or not side["idle"]:', 'if side["ms"] is None:', "timing not idle"),
           ("an unknown fact left in", "report_lib.py", "if v is None:\n            fail(", "if False:\n            fail(", "unknown fact in a text")]


def mutant_verdict(name, rel, old, new, how, tmp):
    """how the mutant is noticed: a short reason, or '' when it survives (a crash of the builder is no reason: a mutant must change a rule, not break the file)"""
    mdir = os.path.join(tmp, "mut_" + name.replace(" ", "_"))
    shutil.copytree(BUILD, os.path.join(mdir, "build"), ignore=shutil.ignore_patterns("__pycache__", "chapters"))
    p = os.path.join(mdir, "build", rel)
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1, f"mutant '{name}': the line to break is not in {rel} exactly once ({s.count(old)})"
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    if how == "out":
        root = os.path.join(mdir, "fx")
        fixture(root)
        rc, msg = build(os.path.join(mdir, "build"), root)
        assert not any(e in msg for e in ("SyntaxError", "NameError", "ImportError", "IndentationError")), f"mutant '{name}' does not run (broken file): {msg[-300:]}"
        return msg.strip().splitlines()[-1][:90] if rc else "; ".join(safe_failures(root))[:90]
    ok, msg = scenario_ok(os.path.join(mdir, "build"), how, mdir)
    assert not any(e in msg for e in ("SyntaxError", "NameError", "ImportError", "IndentationError")), f"mutant '{name}' does not run (broken file): {msg[-300:]}"
    return "" if ok else f"scenario '{how}' no longer stops the build"


def unit():
    import build_report as BR
    import bundle_numbers as BN
    import report_lib as L
    def ck(cond, msg):
        assert cond, msg

    ck(L.compare_key("seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5") == "seg.fov/dt+0.00/lane_divider/all/all/iou@0.5", "compare key of a map row")
    ck(L.compare_key("detf/dt+0.00/vehicle/all/ap@2") == "detf.dt+0.00/vehicle/all/ap@2" and L.compare_key("vel/dt+0.00/cells/static/epe") == "vel.dt+0.00/cells/static/epe", "compare key of det / vel")
    ck(L.compare_key("seg_fov/seg/dt+0.00/centreline:dir/all/all/angle_mae") == "seg.fov/dt+0.00/centreline:dir/all/all/angle_mae", "compare key of the centre-line direction")
    ck(BN.paired_ids("paired scene bootstrap (1000 resamples), b - a; a = runs/c, b = runs/n; val") == ("c", "n"), "ids of the compare header")
    ck(BN.reading(False, ("0", "0", "+0.02", "+0.01", "+0.03", True)) == "better" and BN.reading(True, ("0", "0", "+0.02", "+0.01", "+0.03", True)) == "worse"
       and BN.reading(True, ("0", "0", "-0.02", "-0.03", "-0.01", True)) == "better" and BN.reading(False, ("0", "0", "+0.02", "-0.01", "+0.03", False)) == "not separated"
       and BN.reading(False, None) == "n/a", "the reading rule")
    ck(BR.chapter_html("text", "FULL", "WEAK") == "FULLWEAK<p>text</p>", "a chapter without markers: the numbers, the weak spots, then the text")
    ck(BR.chapter_html("a\n\n@@WEAK@@\n\nb\n\n@@TABLE@@\n", "FULL", "WEAK") == "<p>a</p>WEAK<p>b</p>FULL", "markers in the order of the file")
    ck(L.fill("a @@x@@ b @@y@@", lambda n: {"x": 1, "y": "z"}.get(n)) == "a 1 b z", "fill")
    bank = os.path.join(tempfile.mkdtemp(), "bank.txt")
    open(bank, "w", encoding="utf-8").write("@@@ a\n\nx\ny\n\n@@@ b\nz\n")
    ck(L.read_bank(bank) == {"a": "\nx\ny", "b": "z"}, f"text bank blocks: {L.read_bank(bank)}")
    shutil.rmtree(os.path.dirname(bank))


def scratch_free():
    """the builder and its configs read nothing from a temp/ folder (the inputs are committed in results/build/inputs/), and every file the real configs name exists"""
    import ast
    scratch = re.compile(r"(^|[/\\])temp[/\\]")
    assert scratch.search("temp/t16/x.md") and scratch.search("bev_fusion/temp/t16") and scratch.search("..\\temp\\x") and not scratch.search("results/build/inputs/x") \
        and not scratch.search("template/x"), "the scratch pattern"
    for name in sorted(os.listdir(BUILD)):
        path = os.path.join(BUILD, name)
        if name.endswith(".py"):
            strings = [n.value for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        elif name.startswith("report_runs") and name.endswith(".json"):
            cfg = json.load(open(path, encoding="utf-8"))
            strings, todo = [], [cfg]
            while todo:
                x = todo.pop()
                if isinstance(x, dict):
                    todo += list(x.values())
                elif isinstance(x, list):
                    todo += x
                elif isinstance(x, str):
                    strings.append(x)
            named = list(cfg["text"]) + [v for k, p in cfg["inputs"].items() for v in (p if isinstance(p, list) else [p])] + [c["file"] for c in cfg.get("compare", [])]
            named += [cfg["mail"]["narrative"], cfg["bundle"]["chapter"]] if "bundle" in cfg else [cfg["mail"]["narrative"]]
            missing = [p for p in named if not os.path.isfile(os.path.join(BUILD, "..", "..", p))]
            assert not missing, f"{name}: files named in the config that do not exist: {missing}"
        else:
            continue
        bad = [s for s in strings if scratch.search(s)]
        assert not bad, f"{name} names a path in a temp/ folder (it is scratch and goes away with clean_temp.py): {bad[:3]}"


def real():
    """by hand: the v6 config rebuilds the v6 page; compared with the stored one without what changed since (the commit hash, the architecture drawing) and the mail without BOM / CR"""
    out = tempfile.mkdtemp()
    rc, msg = build_real(out)
    print(msg.strip().splitlines()[-1] if msg.strip() else "")
    norm = lambda s: re.sub(r"main at [0-9a-f]{7}", "main at HEAD", re.sub(r"<svg.*?</svg>", "<svg/>", re.sub(r"<code>[0-9a-f]{7}</code>", "<code>HEAD</code>", s, count=1), flags=re.S))      # noqa: E731
    root = os.path.join(HERE, "..")
    pairs = [("report.html", "results/t16/report_big.html"), ("mail.txt", "results/BEV_result_report/mail.txt")]
    for new, old in pairs:
        a = open(os.path.join(out, new), encoding="utf-8").read()
        b = open(os.path.join(root, old), encoding="utf-8").read().replace("\r\n", "\n").lstrip("﻿")
        print(f"{new} against {old}:", "identical" if norm(a) == norm(b) else "DIFFERENT")
    shutil.rmtree(out)


def build_real(out):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, os.path.join(BUILD, "build_report.py"), "--config", os.path.join(BUILD, "report_runs_v6.json"), "--out", out, "--no-check"], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def main():
    if "--real" in sys.argv:
        return real()
    tmp = tempfile.mkdtemp(prefix="report_build_")
    try:
        unit()
        print("unit: compare keys, the reading rule, markers, fill, the text bank")
        scratch_free()
        print("no path of the code or the configs is in a temp/ folder; every file the configs name exists")
        root = os.path.join(tmp, "good")
        fixture(root)
        rc, msg = build(BUILD, root)
        assert rc == 0, msg
        bad = failures(root)
        assert not bad, "\n".join(bad)
        print("fixture report: numbers in the sources, every cell in place, markers, mail")
        for name in SCENARIOS:
            ok, msg = scenario_ok(BUILD, name, tmp)
            assert ok, f"a build with '{name}' must stop with its message and write no page: {msg[-300:]}"
        print(f"{len(SCENARIOS)} missing / wrong inputs stop the build with a message")
        verdicts = [(m[0], mutant_verdict(*m, tmp)) for m in MUTANTS]
        if "-v" in sys.argv:
            print("\n".join(f"  {n}: {v}" for n, v in verdicts))
        alive = [n for n, v in verdicts if not v]
        assert not alive, f"mutants not noticed: {alive}"
        print(f"{len(MUTANTS)} / {len(MUTANTS)} mutants killed")
        print("report build gate passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
