def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
edit("training/README.md", [("--lr --workers --smoke`,", "--lr --workers --d --layers --heads --ffn --token-drop --none-weight --smoke` (the options of the head and the loss are in the recipe and in `model.pt`; defaults = the first build), "),
                            ("(`trainer`: kill, resume, a fake loader)", "(`trainer`: kill, resume, a fake loader; `arch_options`: defaults = the old head bit for bit, sizes, token drop; `none_weight`)")])
edit("evaluation/README.md", [("`ctl` (c-hi table links", "`score --link-thr T --keep-thr K [--no-none-gate]` (another decode of the same dump, card name `token_card_rec_lt<T>_kt<K>[_nng]..`), `sweep` (the grid of decode variants on the TRAIN sample -> `runs/R/token_sweep_train.json`), `ctl` (c-hi table links"),
                              ("`test_jtoken_eval.py` (E365, CPU, hand frames", "`test_jtoken_eval.py` (E365, CPU, decode variants and sweep, hand frames")])
