#!/bin/bash
cd "$(dirname "$0")/../.."
for f in "$@"; do echo "== $f"; python3 temp/e379/show_card.py "$f"; done
