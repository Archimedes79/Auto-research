"""second round of README rows (read first, write after: the text is computed before the file is opened for writing)"""
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def edit(text, pairs):
    for old, new in pairs:
        assert text.count(old) == 1, (text.count(old), old[:70])
        text = text.replace(old, new)
    return text


def rewrite(path, pairs):
    text = edit(open(path, encoding="utf-8", newline="").read(), pairs)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


rewrite(os.path.join(ROOT, "tests", "README.md"), [
    ("the real tiny model's `sem_1m`. Real cached GT",
     "the real tiny model's `sem_1m`; a hand frame of four GT cells through the real `_BEVSubject.predict` (fake forward) and BevSegMetric: the angle on every meter (fov / seen / out), mass subset and dt with a known answer. Real cached GT"),
    ("the SegMeter counts equal those of a frozen copy of `add_sample` from before the change (4 outputs, 3360 count rows, bit for bit), the Task's spec is the model's own. "
     "13 planted bugs (off-by-one presence channel, no / shifted upsample, direction not upsampled, wrong GT layer, layer missing or added to an old run, no sigmoid, cos / sin swapped, "
     "flipped direction, angle without the wrap, axis angle, zero vector = perfect). CPU, about 70 s",
     "the SegMeter counts and `result()` equal those of frozen copies from before the change (4 outputs, 3360 count rows, bit for bit), the Task's spec is the model's own, the real predict "
     "gives the old output keys. 19 planted bugs (off-by-one presence channel, no / shifted upsample, direction not upsampled, wrong GT layer, layer missing or added to an old run, no sigmoid, "
     "cos / sin swapped, flipped direction, angle without the wrap, axis angle, zero vector = perfect, direction on the fov meter only / unmasked on seen and out / without the splat mass, "
     "a changed result(), predict dropping the layer or the direction). CPU, about 100 s"),
    ("13 / 13 killed", "19 / 19 killed"),
])
rewrite(os.path.join(ROOT, "tests", "run_gates.sh"), [("old runs' counts unchanged, 13 mutants killed; CPU, about 70 s", "old runs' counts unchanged, 19 mutants killed; CPU, about 100 s")])

# evaluation/README.md: the working file (someone else's work in progress; its adapter row still says 12 mutants) and the blob = HEAD + my sentence only
SENT = (" The head's 1 m distance channel is trained but not scored; the oracle's IoU@0.5 is 0.96, not 1.0 (a cell-alignment effect of the 2 x 2 copy: its diagonal cells lie 1.41 cells from the line).")
OLD1 = "(a model without the head: its spec itself, nothing else changes) |"
NEW1 = "(a model without the head: its spec itself, nothing else changes)." + SENT + " |"
rewrite(os.path.join(ROOT, "evaluation", "README.md"), [(OLD1, NEW1), ("old runs unchanged, 12 mutants) |", "old runs unchanged, 19 mutants) |")])
head = subprocess.run(["git", "-C", ROOT, "show", "HEAD:bev_fusion/evaluation/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
head = edit(head, [(OLD1, NEW1), ("old runs unchanged, 13 mutants) |", "old runs unchanged, 19 mutants) |")])
with open(os.path.join(HERE, "README_eval_mine.md"), "w", encoding="utf-8", newline="") as f:
    f.write(head)
print("ok")
