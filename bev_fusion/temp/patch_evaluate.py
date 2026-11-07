"""throw-away: apply the critic's changes to evaluation/evaluate.py (exact-text replacements, each must match once)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
path = os.path.join(ROOT, "evaluation", "evaluate.py")
text = open(path).read()


def sub(old, new):
    global text
    assert text.count(old) == 1, old[:60]
    text = text.replace(old, new)


# docstring
sub('''Global options as in bev_suite: --device cuda|cpu  --workers N  --split val|train  --ckpt-dir DIR  --out DIR. The TEST split is not offered.
Nothing asked for (no option): this help, exit 2.
"""''', '''Global options as in bev_suite: --device cuda|cpu  --workers N  --split val|train  --ckpt-dir DIR  --out DIR. The TEST split is not offered.
Nothing asked for (no option): this help, exit 2.

Refused before any step (exit 2, the message says why): a bad value (--n < 1, a name that is not in the val list: `bev_suite.py RUN list` is read for it, about a minute),
no checkpoint, and outputs of the run that a step would rewrite and that exist already (eval/card.json for --quant, eval/timing_modules.json and latency.json for --time:
the delivered runs are not overwritten by accident; --replace does it on purpose, or evaluate a copy of the run folder).
Exit code: 0 when every step passed; otherwise the exit code of the first step that failed (1 if it gave none). A failed step of --all other than the numbers does not stop
the chain, but the call still ends non-zero, with one summary line of what failed; RUN/eval/door.json keeps every step.
"""''')
# imports
sub("import argparse\nimport json\n", "import argparse\nimport glob\nimport json\n")
# parser
sub('''    p.add_argument("--n", type=int, default=1000, help="val frames of --quant / --all (default 1000)")''',
    '''    p.add_argument("--n", type=positive_int, default=1000, help="val frames of --quant / --all (default 1000)")''')
sub('''    p.add_argument("--out", default=None, help="the suite folder (default RUN/eval/suite)")
    return p
''', '''    p.add_argument("--out", default=None, help="the suite folder (default RUN/eval/suite); timing_modules.json goes into it too")
    p.add_argument("--replace", action="store_true", help="rewrite outputs of RUN/eval that exist already (card.json, timing_modules.json, latency.json)")
    return p
''')
sub('''def build_parser() -> argparse.ArgumentParser:''', '''def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text}: at least 1 (bev_suite's `eval --n 0` = all frames is not offered here)")
    return n


def build_parser() -> argparse.ArgumentParser:''')
# messages without the prefix (main adds it)
sub('raise SystemExit(f"evaluate: --{what} {text}: at least 1")', 'raise SystemExit(f"--{what} {text}: at least 1")')
sub('raise SystemExit(f"evaluate: --{what} {text!r}: a number or a comma list of "', 'raise SystemExit(f"--{what} {text!r}: a number or a comma list of "')
sub('raise SystemExit("evaluate: --all with --split train:', 'raise SystemExit("--all with --split train:')
# global options as a function, timing out file
sub('''
    def global_options(out=a.out):                                                  # the global options of bev_suite, always written out
        return (["--device", a.device, "--workers", str(a.workers), "--split", a.split] + (["--ckpt-dir", a.ckpt_dir] if a.ckpt_dir else [])
                + (["--out", out] if out else []))
    G, soft = global_options(), a.all                                               # soft: in the chain only the numbers are fatal
''', '''    G, soft = global_options(a, a.out), a.all                                       # soft: in the chain only the numbers are fatal
''')
sub('''        steps.append(step("timing", G + ["timing", "--n", str(TIMING_N)], fatal=not soft))''',
    '''        into = os.path.join(a.out, "timing_modules.json") if a.out else None        # bev_timing's --out is a FILE (bev_suite hands its --out on), the suite's is a folder
        steps.append(step("timing", global_options(a, into) + ["timing", "--n", str(TIMING_N)], fatal=not soft))''')
sub('step("showcase", global_options(show) + [', 'step("showcase", global_options(a, show) + [')
sub('''def plan(a) -> list:''', '''def global_options(a, out=None) -> list:
    """the global options of bev_suite, always written out; out = the --out to hand on (None: none)"""
    return (["--device", a.device, "--workers", str(a.workers), "--split", a.split] + (["--ckpt-dir", a.ckpt_dir] if a.ckpt_dir else [])
            + (["--out", out] if out else []))


def plan(a) -> list:''')
open(path, "w").write(text)
print("ok")
