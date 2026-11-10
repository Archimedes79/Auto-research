import sys

mode, path = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


if mode == "eval":
    rep('''old `eval` without `--warm` starts every frame cold), and the predictions are stored, so a new GT is scored without the network. The val loss of a result is the one computed when the network
ran (it needs the ground truth of that moment): it is kept as it was, whatever GT a later scoring uses.''', '''old `eval` without `--warm` starts every frame cold), the predictions are stored, so a new GT is scored without the network, and the network is run by the Runner (D72), which reads no ground
truth: a prediction carries no validation loss, so `loss.*` / `val_loss/*` are not among the numbers of an Evaluation (`bev_suite.py eval` keeps them).''')
elif mode == "readme":
    rep('''as a warning. What the interface does not do yet: a ground-truth-free `predict` (the Subject computes its val loss from the GT, so a frame without labels needs a Subject without the loss: a change
of the frozen `bev_adapter.py` or a second Subject class). The val loss is computed by the network while it
predicts, against the ground truth of that moment: a rescoring under another GT keeps it and says so (`warnings`; the card leaves `val_loss/*` out). `run` here''', '''as a warning. The network is run by the `Runner` class (`Runner.run`, `check`, `tap`, `time`, `probe`; D72), which reads no ground truth and computes no validation loss: it works alone on frames
without labels (the BEV side: `bev_free.py`, `free_source` / `free_subject` of the adapter), `Evaluation.run` calls it and scores its predictions; so an Evaluation has no `loss.*` / `val_loss/*`
(`bev_suite.py eval` keeps them). `run` here''')
open(path, "w", encoding="utf-8").write(s)
print("ok")
