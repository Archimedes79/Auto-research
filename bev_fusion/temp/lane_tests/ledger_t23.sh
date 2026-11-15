#!/bin/bash
# scratch: write the judgement columns of the t23 arm C row (the facts are written by run.py)
cd "$(dirname "$0")/../../.." || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
python3 autoresearch/tools/ledger.py note t23.C_s0 \
  --result "VAL 1000, dt 0, one seed, against t21.ctl_s0: lane divider IoU@0.5 0.4341 -> 0.4287 (-0.0054, paired 95% [-0.0135, +0.0022]); IoU@1 0.5516 -> 0.5601 (+0.0086); AP 0.2149 -> 0.2533; roc_auc 0.8385 -> 0.9281; f1_max 0.3222 -> 0.3328, best-F1 threshold 0.902 -> 0.769; prediction (IoU@0.5 >= 0.455, IoU@1 >= 0.565) missed; guards hold within m except walkway IoU@0 -0.0102 (m 0.010, interval excludes 0) and vehicle IoU@0 -0.0020; lane-centre rows within 0.003; step-1 check equal; no NaN" \
  --decision "TIE by hand: composed from the table rows (card section 1): the IoU@0.5 difference is inside +-m (m 0.010); provisional for one seed and for these settings (weight 10, positives x3, ignore 1.0 m); ranking much better but the 0.5 operating point is not; no second seed started"
