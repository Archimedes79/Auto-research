#!/bin/bash
# scratch: RunJobs -> Runner in the files that name it (the consolidated runsuite.Runner)
cd /mnt/c/Auto-research/autoresearch || exit 1
for f in trainsuite/selftest_api.py trainsuite/README.md trainsuite/api.py README.md; do
  sed -i 's/RunJobsError/RunnerError/g; s/RunJobs/Runner/g' "$f" || exit 1
done
grep -n "RunJobs" -r . --include=*.py --include=*.md | head
echo "left: $(grep -rn 'RunJobs' . --include=*.py --include=*.md | wc -l)"
