"""One-off: oracle (no learning) recovered share on train vs held-out scenes, at the nominal and the tight (x0.25) tolerances.
A train - held-out difference of the ORACLE is a property of the data, not overfitting.
Run: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/oracle_split_gap.py"""
import sys
sys.path.insert(0, "tests")
import radar_codecs as rc
rt = rc.rt
tr, te = rt.load(set(range(8))), rt.load({8, 9})
c = rc.Codec("abs")
for scale in (1.0, 0.25):
    if scale != 1.0:
        for k in rt.TOL:
            rt.TOL[k] *= scale
        rc.R_TOL, rc.PSI_TOL, rc.DOP_TOL, rc.RCS_TOL, rc.AGE_TOL, rc.OFF_TOL = (v * scale for v in (0.5, 0.0523599, 0.5, 3.0, 0.1, 0.5))
        c = rc.Codec("abs")
    a = rc.eval_codec(c, None, tr)[1].float().mean().item()
    b = rc.eval_codec(c, None, te)[1].float().mean().item()
    print(f"tolerance x {scale}: oracle train {100 * a:.1f}%, held-out {100 * b:.1f}%, difference {100 * (a - b):+.1f}")
