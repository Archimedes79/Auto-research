#!/bin/bash
# Move finished one-off scripts (exact-string patch scripts, dry runs of finished tests, a stale generated model copy) of
# bev_fusion/temp/ into _to_delete/temp_oneoffs/ (nothing is deleted, git mv keeps the history).
# A script is moved only when it is tracked, unmodified, and no other file of the repository mentions its name.
cd "$(dirname "$0")/../.." || exit 1
dest=_to_delete/temp_oneoffs
mkdir -p $dest
cands="temp/checks/plan_*.py temp/checks/edit_*.py temp/checks/dry_*.py temp/study/patch_*.py temp/setup/replace_section2.py
temp/setup/replace_section6.py temp/setup/fix_f4_axes.py temp/setup/fix_extract_prefix.py temp/checks/make_model_mine.py
temp/checks/run_model_mine.py"
dirs="temp/radar/_scratch_patches temp/checks/model_mine"
moved=0; kept=0
for f in $cands $dirs; do
  [ -e "$f" ] || continue
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || git ls-files "$f" | grep -q . || { echo "skip (untracked) $f"; continue; }
  [ -z "$(git status --short -- "$f")" ] || { echo "skip (modified) $f"; kept=$((kept+1)); continue; }
  b=$(basename "$f" .py); b=${b%.sh}
  refs=$(git grep -l -w -- "$b" -- . ':!temp/review' ':!_to_delete' | grep -v "^$f" | grep -v "^temp/checks/plan_\|^temp/checks/edit_\|^temp/checks/dry_\|^temp/study/patch_" | head -3)
  if [ -n "$refs" ]; then echo "skip (referenced by $(echo $refs | tr '\n' ' ')) $f"; kept=$((kept+1)); continue; fi
  tgt=$dest/$(echo "$f" | sed 's#^temp/##; s#/#__#g')
  git mv "$f" "$tgt" && moved=$((moved+1))
done
echo "moved $moved, kept $kept -> $dest"
