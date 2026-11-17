p = "experiments/E312_combined-end-maps/experiment.md"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub("then the E305 pair head (maps and pairs trained together) on it, against the E305 pair head on the maps of E286 j1; delivered with snapped connector ends at p >= 0.1 (E311)\"",
    "then the E305 pair head (maps and pairs trained together) with the learned handle scales of E303 on it, against the E305 pair head on the maps of E286 j1; delivered with snapped connector ends at p >= 0.1 (E311)\"")
sub("then `jends_train.py --e2e --maps-w 1.0 --steps 3000` on them (run e312_p_s<S>, BFLC_SIG=1.0 so that the heat targets of the joint training match the maps); same read-out\"",
    "then `jends_train.py --e2e --scales --maps-w 1.0 --steps 3000` on them (run e312_p_s<S>, BFLC_SIG=1.0 so that the heat targets of the joint training match the maps); the learned handle scales are used by the read-out; same read-out as the control\"")
sub("## 1. Order of work", "**Amendment 2026-11-12 11:25 (before any run of this card):** the pair head gets the handle scales of E303 (`--scales`). On the snapped read-out the scales are worth more than on the unsnapped one: E303 seed 1, same pairs, p >= 0.1: topo F1 +0.0356 with the learned scales against +0.0251 with the chord tangent (F1@1.0 +0.0363 against +0.0298); the seed-0 control was still being scored when this amendment was written. The control c stays E305 (no scales), so the arm changes TWO things (maps m2, scales); the reference points E303 (scales, frozen maps: +0.0320 / +0.0356 over the plain graph) and E305 (e2e, no scales: +0.0304 / +0.0285) are given in the result so that the two effects can be read apart. The scales make the pair training slower (the fit of the GT scales: 1.9 s / step instead of 1.2).\n\n## 1. Order of work")
open(p, "w", encoding="utf-8").write(s)
p = "temp/e312/start_pairs.sh"
s = open(p, encoding="utf-8").read()
sub("# E312 pair head on the m2 maps (maps and pairs trained together, the recipe of E305)", "# E312 pair head on the m2 maps (maps and pairs trained together, the recipe of E305, + the learned handle scales of E303)")
sub('--hyp "The E305 pair head (maps and pair head trained together) on the narrower, far-weighted maps m2 delivers a better snapped read-out than on the maps of E286 j1."', '--hyp "The E305 pair head (maps and pair head trained together) with the E303 handle scales on the narrower, far-weighted maps m2 delivers a better snapped read-out than the E305 head on the maps of E286 j1."')
sub("--e2e --maps-w 1.0 --steps 3000 --workers 3", "--e2e --scales --maps-w 1.0 --steps 3000 --workers 3")
open(p, "w", encoding="utf-8").write(s)
print("ok")
