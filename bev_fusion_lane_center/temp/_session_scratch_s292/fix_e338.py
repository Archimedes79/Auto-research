p = "experiments/E338_stack-check-e324s-on-c2-graph/experiment.md"
s = open(p, encoding="utf-8").read()
i0 = s.index(' "kill": "invalid (not a verdict)')
i1 = s.index('\n "evaluate"')
new = (' "kill": "invalid (not a verdict): the scorer\'s pyramid assert fails, or the CONTROL does not reproduce: the same tagged scoring of e324_s_s<S> on B\'s own dump (`--snap --thr 0.1 --full temp/e276/predfull_e273_p_s<S>.pkl --base runs/e273_p_s<S>/model.pt --spec close:1.5@0.5 --tag e338b`) must reproduce the stored card of s (`ends_card_snap_thr0.1.json`) to 1e-6 (the plumbing was verified on E312 s0 on 2026-11-14 with its card of record; this checks it for the E324 s pair run)",')
s = s[:i0] + new + s[i1:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
