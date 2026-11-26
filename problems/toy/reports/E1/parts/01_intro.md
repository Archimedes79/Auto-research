**Question.** Does a network with 12 channels mark the disc better than one with 4, measured as the pixel IoU on the val split?

**Rule, written before the runs.** Keep the wider network when its IoU is higher by at least 0.010 and its recall is not lower by more than 0.020, on two seeds.

**Verdict by that rule: KEEP.** The effect is {{effect}}, the paired scene intervals of the two seeds span {{lo}} to {{hi}}. The prediction was +0.020 and missed by far.

**What it does not show.** The narrow control did not finish training in the given steps (IoU {{iou_c0}} for seed 0, {{iou_c1}} for seed 1). The experiment therefore measures how far each network gets in this budget, not what the number of channels can hold. The kill rule watched the training loss, which falls while the predicted mask stays empty, so it did not fire.
