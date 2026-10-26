#!/usr/bin/env bash
# One-off: write runs/<name>/code/ from a pinned commit for a run that was started without a snapshot (t1_r0: trained with
# the working tree at 914ec77; 2ddf511 adds only opt-in options with unchanged defaults), so evaluate_run uses that code.
cd "$(dirname "$0")/../.." || exit 1
name=$1; rev=$2; note=$3
mkdir -p runs/$name/code
git -C .. archive "$rev" bev_fusion/training | tar -x -C runs/$name/code --strip-components=2 || exit 1
echo "commit $(git rev-parse --short "$rev") (git archive of bev_fusion/training); $note" > runs/$name/code/SNAPSHOT.txt
cat runs/$name/code/SNAPSHOT.txt
