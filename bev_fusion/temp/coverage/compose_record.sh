#!/bin/bash
# the record of the footprint-share study = head + the tables of the analysis report + tail
cd "$(dirname "$0")/../.." || exit 1
{
  cat temp/coverage/rec_head.md
  echo
  echo "## The tables (made from the object list of the extraction; f = the share inside the fusion area, all objects with a part inside the grid)"
  echo
  sed '1d' temp/coverage/footprint_share_report_val_f_fus.md | sed 's/^## /### /'
  echo
  cat temp/coverage/rec_tail.md
} > records/footprint_share_study_2026-11-01.md
wc -c records/footprint_share_study_2026-11-01.md
