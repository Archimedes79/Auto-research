cd /mnt/c/Auto-research
python3 autoresearch/tools/commit_paths.py -m "handover 17:00 (final state of 2026-11-12)" --trailer "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" bev_fusion_lane_center/HANDOVER.md 2>&1 | tail -n 1
git status --short | grep -v "^??" | head -10
python3 autoresearch/tools/check_git.py 2>&1 | grep -E "pending|unpushed|ahead" | head -4
