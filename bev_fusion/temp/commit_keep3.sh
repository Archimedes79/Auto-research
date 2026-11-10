#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/records/keep3.json"
cat > bev_fusion/temp/commit_keep3_msg.txt <<'EOF'
keep3: the best three re-picked on GT v8 (Mirko 2026-11-10: "Ja, neu waehlen"): t21_ctl (board7 0.4549, two seeds), t23_C (0.4497, two seeds), t20_base (0.4465, the accepted reference baseline); the nine best arms lie within 0.011 (band 0.02), so the slots go by role; the three of 2026-11-01 (t11_vel_hurdle_acc2, t10_dual, t8_ctl: pre-D48, GT 15efbde2d1, cannot run through the Runner) are in "left" with their checkpoint paths (nothing deleted, the old record is in git history); all three new arms ran through the Runner on real frames

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/commit_keep3_msg.txt $P || exit 1
git log --oneline -1 | cut -c1-140
