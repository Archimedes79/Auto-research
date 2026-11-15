def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
sub("tests/README.md",
    "compared: log rows, bitwise weights, EMA, optimiser, RNG, scaler, files, messages, exit codes; 8 mutants of trainkit are killed |",
    "compared: log rows, bitwise weights, EMA, optimiser, RNG, scaler, files, messages, exit codes (the TrainConfig fields added after that commit, `ADDED` in the test = the three t22 anchor fields, default off, are left out of the configs compared); 8 mutants of trainkit are killed |")
