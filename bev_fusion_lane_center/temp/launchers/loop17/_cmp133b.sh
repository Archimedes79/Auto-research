#!/bin/bash
BFLC="${BFLC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"; REPO="$(dirname "$BFLC")"
S=$REPO/scripts
bash $S/_py14.sh p17_merge.py --prefix e133b --md tables/e133b.md 2>&1 | tail -30
