#!/bin/bash
# Greatest set D of orphan files that nothing live names: start with all orphans, drop any file named by a tracked CODE file outside D, repeat.
# Referrers = tracked .py/.sh/.json files outside temp/, experiments/*/code, records; docs (md) and architecture.json are not referrers (rows are removed with the file).
cd "$(dirname "$0")/../../.." || exit 1          # repo root
L=bev_fusion_lane_center
mapfile -t D < <(sed "s#^#$L/#" "$L/temp/e330/orphans.txt")
changed=1
while [ $changed = 1 ]; do
  changed=0
  NEW=()
  for f in "${D[@]}"; do
    s=$(basename "$f"); s=${s%.*}
    keep=0
    while read -r q; do
      [ "$q" = "$f" ] && continue
      inD=0; for g in "${D[@]}"; do [ "$g" = "$q" ] && inD=1 && break; done
      if [ $inD = 0 ]; then keep=1; echo "KEEP $f  <- $q" >> "$L/temp/e330/closure_why.txt"; break; fi
    done < <(git grep -lw -- "$s" -- "$L/*.py" "$L/*.sh" "$L/*.json" "autoresearch/*.py" "autoresearch/*.sh" "GtGenerator/*.py" "GtGenerator/*.sh" "bev_fusion/*.py" "bev_fusion/*.sh" \
        ":(exclude)$L/temp" ":(exclude)$L/experiments" ":(exclude)$L/experiment_execution" ":(exclude)$L/runs" ":(exclude)$L/results" ":(exclude)$L/reports" ":(exclude)$L/literature" ":(exclude)$L/training/architecture.json" ":(exclude)$L/data/rebuild.json_NOT" 2>/dev/null)
    if [ $keep = 1 ]; then changed=1; else NEW+=("$f"); fi
  done
  D=("${NEW[@]}")
done
printf '%s\n' "${D[@]}" > "$L/temp/e330/delete_set.txt"
echo "delete set: ${#D[@]} files"
