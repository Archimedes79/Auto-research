#!/bin/bash
# scratch: commit the clean-base round of the Runner area through a temporary index (own paths; tests/README.md as a blob built from the CURRENT HEAD)
cd /mnt/c/Auto-research || exit 1
python3 bev_fusion/temp/blobs_round.py || exit 1
cat > bev_fusion/temp/commit_round_msg.txt <<'EOF'
runner clean-base round: Runner.predict makes the result set at its first frame (a refused run - unknown tap, tap the network does not produce, network too old, changed checkpoint - leaves none, and the lock is taken before anything is written), plan builds the subject and finds what predict refuses, a shorter look at a scene keeps the memory warning a longer run recorded (non-finite fill is a warning too), memory_checked says whether the memory could be judged, the changed-checkpoint error names the stored id; list_results carries kind; runner/README says which networks run (the three of keep3.json are pre-D48 and are refused with the reason); tests: warnings, refused run, health asserted, run probe of the conformance test skips with a message without runs, real test compares ids only when the prediction code was not edited meanwhile and uses the project's Evaluation; records/jobs READMEs name the final classes, the dead example job file t20_val8.json (its job layer is gone) leaves; no 'frozen' wording

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
P="bev_fusion/runner/runner.py bev_fusion/runner/__init__.py bev_fusion/runner/README.md bev_fusion/README.md \
bev_fusion/tests/test_runner_class.py bev_fusion/tests/test_runner_real.py bev_fusion/tests/test_suites_conform.py \
bev_fusion/records/jobs/README.md bev_fusion/records/jobs/running/README.md bev_fusion/records/jobs/evaluation/README.md bev_fusion/records/jobs/running/t20_val8.json"
git status --short -- $P bev_fusion/tests/README.md | cat
python3 autoresearch/tools/check_git.py --paths $P bev_fusion/tests/README.md || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/commit_round_msg.txt $P \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/blob_round_tests_README.md || exit 1
git log --oneline -1 | cut -c1-140
git status --short -- $P | cat
