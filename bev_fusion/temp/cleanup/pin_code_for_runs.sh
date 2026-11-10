#!/bin/bash
# scratch (Image Encoder, clean-base round, the lead's order 2026-11-10): give the three stored runs without a code/ folder a pinned copy of the code of their tag, next to the run's files (nothing overwritten):
#   runs/<run>/code/  = git archive <tag> bev_fusion/training  (+ trainkit = git archive <tag> autoresearch/trainkit, as every launcher snapshot has) + a SNAPSHOT.txt that says where it came from
# smoke_t22_all_cpu has no tag of its own: it was made on 2026-11-07 20:31 on the live tree, the same code era as bf/t22.B_all; that tag is used and the SNAPSHOT.txt says so.
cd /mnt/c/Auto-research || exit 1
pin() {   # pin <run> <tag> <note>
  local run=bev_fusion/runs/$1 tag=$2 note="$3"
  if [ -e $run/code ]; then echo "SKIP $1: code/ exists"; return; fi
  [ -d $run ] || { echo "SKIP $1: no run folder"; return; }
  git rev-parse -q --verify "refs/tags/$tag" > /dev/null || { echo "SKIP $1: tag $tag missing"; return; }
  mkdir $run/code || return
  git archive $tag bev_fusion/training | tar -x -C $run/code --strip-components=2
  git archive $tag autoresearch/trainkit | tar -x -C $run/code --strip-components=1
  printf 'commit %s (git archive of bev_fusion/training and autoresearch/trainkit; tag %s)\nadded 2026-11-10 by the clean-base round (BEV system lead): this run had no code/ folder and imported the live training/, which no longer holds the t22 student_frames "all" option (commit e7366b23ec); %s\n' \
    "$(git rev-parse --short=10 $tag^{commit})" "$tag" "$note" > $run/code/SNAPSHOT.txt
  echo "PINNED $1 <- $tag: $(ls $run/code | wc -l) entries in code/, $(find $run/code -type f | wc -l) files"
}
pin t22_B_all bf/t22.B_all "the stage-B checkpoint of runs/t22_B_s0 evaluated with student_frames all (gate C0, diagnostic)"
pin t22_standin_all_pair4 bf/t22.standin_all_pair4 "the stand-in of gate C0 (frozen A2 student on both frames in the checkpoint of runs/t21_pair4_s0)"
pin smoke_t22_all_cpu bf/t22.B_all "CPU smoke of the stand-in with student_frames all (2026-11-07 20:31); no tag of its own, the tag of the same night is used"
