#!/bin/bash
# the two scratch trees were tagged and moved by clean_temp.py (bf/speed-scratch, bf/speed_base-scratch; the tag keeps the scripts). The other files are copies exported from commit c7b2665:
# delete the moved copies only if every non-script file equals its committed version byte for byte (.pyc caches aside) and no symlink is left
cd /mnt/c/Auto-research/bev_fusion || exit 1
ok=1
for pair in "speed:temp_speed/speed" "speed_base:temp_speed_base/speed_base"; do
  name=${pair%%:*}; dir=_to_delete/${pair##*:}
  [ -d "$dir" ] || { echo "$dir: not there"; continue; }
  links=$(find "$dir" -type l | wc -l)
  git -C .. ls-tree -r --name-only bf/$name-scratch | sed -n "s#^bev_fusion/temp/$name/##p" | sort > /tmp/tag_$name.txt
  (cd "$dir" && find . -type f | sed 's#^\./##' | sort) > /tmp/dir_$name.txt
  comm -13 /tmp/tag_$name.txt /tmp/dir_$name.txt > /tmp/rest_$name.txt
  bad=0; checked=0; pyc=0
  while read -r f; do
    case "$f" in *.pyc) pyc=$((pyc+1)); continue;; esac
    checked=$((checked+1))
    if ! git -C .. show "c7b2665:$f" 2>/dev/null | cmp -s - "$dir/$f"; then echo "  differs from c7b2665: $f"; bad=$((bad+1)); fi
  done < /tmp/rest_$name.txt
  echo "$dir: $links symlinks, not in tag: $checked files compared with c7b2665 ($bad differ), $pyc .pyc caches"
  [ "$links" = 0 ] && [ "$bad" = 0 ] || ok=0
done
if [ "$ok" = 1 ]; then
  rm -rf _to_delete/temp_speed _to_delete/temp_speed_base && echo "deleted the two moved copies"
else
  echo "NOT deleted"
fi
ls _to_delete
