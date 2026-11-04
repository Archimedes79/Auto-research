#!/bin/bash
# scratch: commit of the deletion of the 174 uncited bev_fusion paper notes (Mirko: "Delete 181 notes", D49)
cd /mnt/c/Auto-research || exit 1
P=$(cat bev_fusion/temp/t16/commit_paths_lit.txt | tr '\n' ' ')
python3 autoresearch/tools/check_git.py --paths $P 2>&1 | tail -n 3
cat > bev_fusion/temp/t16/msg_lit.txt <<'EOF'
bev_fusion literature: 174 paper notes that nothing live cites removed (Mirko: "Delete 181 notes", D49)

The documentation review of 2026-11-03 found 216 notes in bev_fusion/literature/papers/, read by no program: 42 are cited by code, cards or the SOA digest (they stay), 174 are cited by nothing live (the review said 181; the 7 more are the sources of published numbers in the SOA digest and stay). papers.csv, literature/code/ and the two scouts stay. Every removed note comes back with git checkout archive/docs-before-cleanup-2026-11-03 -- bev_fusion/literature/papers/<file>. plan/plan.md still cites some of them and is to be trimmed in place (frozen files cite its path). README row and the D49 / review records updated.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/t16/msg_lit.txt $P 2>&1 | tail -n 4
git log --oneline -2 | cut -c1-120
git show --stat --format=%h HEAD | tail -n 2
