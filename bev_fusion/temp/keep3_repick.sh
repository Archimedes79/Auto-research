#!/bin/bash
# scratch: re-pick the best three on the new GT (Mirko 2026-11-10: "Ja, neu waehlen"): back up the old record, init on GT v8, three by-hand considers, the old three into "left"
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
K=bev_fusion/records/keep3.json
[ -f bev_fusion/temp/keep3_old_record.json ] || cp $K bev_fusion/temp/keep3_old_record.json
rm -f $K
KEYS="seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0.5,seg_fov/seg/dt+0.00/walkway/all/all/iou@0,seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0,seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0,seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5,seg_fov/seg/dt+0.00/road_divider/all/all/iou@1,det/dt+0.00/mean/all/mAP"
python3 autoresearch/tools/keep3.py init --source arms --metric board7 --split val --frames 0 --slots 3 --band 0.02 --metric-keys "$KEYS" --gt-signature 0.5_4_6a7aa7864a --ckpt "runs/{run}/ckpt_*.pt" --card "runs/{name}/eval/metrics.json" || exit 1
BASE="by hand: 2026-11-10, Mirko in chat: 'Ja, neu waehlen' (the three of 2026-11-01 are pre-D48 and cannot run through the Runner, the new line is on GT v8 0.5_4_6a7aa7864a). board7 on val: the nine best arms lie within 0.011 (band 0.02), a tie by the metric, so one slot each by role, two-seed arms first:"
python3 autoresearch/tools/keep3.py consider t21_ctl --by-hand "$BASE t21_ctl = the best arm by value (0.4549, two seeds), the control of the temporal-fusion line" --apply || exit 1
python3 autoresearch/tools/keep3.py consider t23_C --by-hand "$BASE t23_C = the newest recipe (tolerance-aware lane BCE) on two seeds (0.4497, TIE with its control by the card)" --apply || exit 1
python3 autoresearch/tools/keep3.py consider t20_base --by-hand "$BASE t20_base = the accepted reference baseline of the line (D48; 0.4465, one seed), the network the Runner tests and the comparisons stand on" --apply || exit 1
python3 - <<'EOF'
import json
new = json.load(open("/mnt/c/Auto-research/bev_fusion/records/keep3.json"))
old = json.load(open("/mnt/c/Auto-research/bev_fusion/temp/keep3_old_record.json"))
for e in old["entries"]:
    new.setdefault("left", []).append(dict(name=e["name"], key=e["key"], value=e["value"], left="2026-11-10", replaced_by="re-pick on GT v8 (Mirko 2026-11-10: 'Ja, neu waehlen')", checkpoints=[c["path"] for c in e["checkpoints"]], gt_signature=old["gt_signature"]))
new["left"] = old.get("left", []) + new["left"] if old.get("left") else new["left"]
json.dump(new, open("/mnt/c/Auto-research/bev_fusion/records/keep3.json", "w"), indent=1)
print("left:", [(x["name"], x["value"]) for x in new["left"]])
EOF
python3 autoresearch/tools/keep3.py show 2>&1 | cut -c1-260
