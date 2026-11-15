"""scratch: the velocity-complete base card is in; the GT shift of the base is stated in full; the base arm lists its card"""
P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep("the headline rows do not move with the GT version; the velocity row was not in that evaluation (no `--velocity`) and is made by `t26.base_eval_vel` (the velocity rows are where v13 differs: stationary flag, flow).",
    "the headline rows do not move with the GT version. **With the velocity rows (`t26.base_eval_vel`, same 1000 frames, `--curves --fusion --velocity`; the card is kept as `runs/t26_base/eval/card.json`):** centre-field EPE of moving objects 1.1037 -> 1.1037 (the stored value 1.103726, now 1.103738), `detf` mAP 0.40920 -> 0.40924, lane divider IoU@0.5 0.42564 -> 0.42566; of the 7,446 numeric rows the two cards share, the only large changes are the cell counts of the centre-line rows (`n_gt` up by 0.4 % = up to 3,015 cells: the lane GT l5 -> l6), the metric values move by at most the CPU / GPU float noise of 4e-5; 1,512 rows of the stored card (other evaluation flags) are not in the new one and none of them is a clause of this card. The GT version does not change this base's control values.")
rep('"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base", "t26.base_eval_vel"]}},',
    '"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base", "t26.base_eval_vel"], "cards": ["runs/t26_base/eval/card.json"]}},')
open(P, "w").write(s)
print("ok")
