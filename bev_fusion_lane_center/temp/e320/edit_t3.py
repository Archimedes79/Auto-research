p = "temp/e320/t_real.py"
s = open(p, encoding="utf-8").read()
old = '''    sys.argv = ["jends_train.py", "--name", "e320_smoke_train_cpu_s0", "--seed", "0", "--maps", os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0", "--geom-w", "0.5", "--attach", "--bs", "2", "--workers", "0", "--smoke"]
    M["jends_train"].main()'''
new = '''    sys.argv = ["jends_train.py", "--name", "e320_smoke_train_cpu_s0", "--seed", "0", "--maps", os.path.join(ROOT, "runs", "e307_n_s0", "model.pt"), "--e2e", "--scales", "--maps-w", "1.0", "--geom-w", "0.5", "--attach", "--bs", "1", "--workers", "0", "--smoke"]
    TT = M["jends_train"].T

    def fake_fit(model, loss_fn, run, *, steps, loader, **kw):                                         # the trainer needs a GPU: call the loss function of main() 101 times on the CPU (no backward) to run the step-100 log line
        n = 0
        with torch.no_grad():
            for b in loader:
                loss_fn(model, b)
                n += 1
                if n >= 101:
                    break
    TT.fit = fake_fit
    M["jends_train"].main()
    print(open(os.path.join(ROOT, "runs", "e320_smoke_train_cpu_s0", "train.log")).read()[-900:])'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
