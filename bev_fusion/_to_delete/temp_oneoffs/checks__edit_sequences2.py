"""one-off: --sequences must not load the training data (move its branch before the data loading)"""
p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''    n_train, n_test, iters = (300, 16, 600) if QUICK else (1500, 48, 3000)
    if DATA == "nuscenes":''', '''    n_train, n_test, iters = (300, 16, 600) if QUICK else (1500, 48, 3000)
    saved = f"temp/checks/model_explains{TAG}.pt"                    # scratch (not in git): the trained test model
    if "--sequences" in sys.argv:                                     # evaluation only, on the model a run saved
        model = Model().to(DEV)
        model.load_state_dict(torch.load(saved, map_location=DEV))
        model.eval()
        seqs = nuscenes_sequences(True, arg("--n-seq", 24), arg("--length", 5))
        sequence_report(model, seqs, f"temp/checks/flow_sequences{TAG}.png")
        sys.exit(0)
    if DATA == "nuscenes":''')
rep('''    saved = f"temp/checks/model_explains{TAG}.pt"                    # scratch (not in git): the trained test model
    if "--sequences" in sys.argv:                                     # evaluation only, on the model a run saved
        model.load_state_dict(torch.load(saved, map_location=DEV))
        model.eval()
        seqs = nuscenes_sequences(True, arg("--n-seq", 24), arg("--length", 5))
        sequence_report(model, seqs, f"temp/checks/flow_sequences{TAG}.png")
        sys.exit(0)
    train(model, data, iters, random.Random(SEED + 7))''', '''    train(model, data, iters, random.Random(SEED + 7))''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
