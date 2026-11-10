def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/README.md", [
    ('''python -m evalsuite Runner.run runs/t23_C_s1 small.val --taps fused           # predictions only, no ground truth (the Runner's words always carry `Runner.`; a bare word is the Evaluation's)
''', '''python -m evalsuite tap t23_C_s1@r3f9a2c1d4b0 scene-0061/12 --name fused       # an intermediate result stored with the predictions (asked for with `run --taps fused`)
'''),
    ('''(`Runner.run` and `Evaluation.score` are the two halves of `Evaluation.run`)''', '''(`Runner.run` and `Evaluation.score` are the two halves of `Evaluation.run`; the Evaluation is the front door, the command line offers only it)'''),
    ('''read with `Runner.tap`;''', '''read with `Evaluation.tap`;'''),
    ('''The network is run by the `Runner` class (`Runner.run`, `check`, `tap`, `time`, `probe`; D72), which reads no ground truth and computes no validation loss: it works alone on frames
without labels (the BEV side: `bev_free.py`, `free_source` / `free_subject` of the adapter), `Evaluation.run` calls it and scores its predictions;''', '''The network is run by the `Runner` class (the layer below the Evaluation: `run`, `check`, `tap`, `time`, `probe`; D72; its own command line is runsuite's), which reads no ground truth and computes no
validation loss: it works alone on frames without labels (the BEV side: `bev_free.py`, `free_source` / `free_subject` of the adapter), `Evaluation.run` calls it and scores its predictions;'''),
])
print("ok")
