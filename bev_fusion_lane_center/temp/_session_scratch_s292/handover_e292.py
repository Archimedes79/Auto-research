p = "HANDOVER.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("3. **E292**"))
L[i] = ("3. **E292** card v3 (critic REVISE answered). CODE BUILT 2026-11-13 22:50, CPU-tested, UNCOMMITTED until the code critic is answered (card section 6 / 7): `QGraph(soft_nms=['mnms', R, sigma])`, phase 9 (pyramid frozen), `--soft-nms-mnms`, "
        "log-space `qmnms` (the old 1e-6 clamp killed the gradient beyond |logit| 13.8), scorer `--full --base --spec --tag` (tagged cards never overwritten), `training/grid/test_mnms_layer.py [--real]` (kill criterion not triggered: 2e-7 on 20 real frames). "
        "Run: `temp/e292/start_arm.sh <c2|a> <seed>`, `post_arm.sh`, `pre_check.py`; chain `temp/day15/day15.sh` waits for `temp/e292/READY` + `ACCEPTED` (written after the critic) and for the E324 arm-s training, then c2 (both seeds), the precondition, then a. Flag-off identity: `bash tools/env/_py14.sh ../temp/s292/ref_b.py check`.")
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("ok")
