#!/bin/bash
# runs the mutation script: the 29 new mutants first, then the 19 old ones (scratch, one-time use)
cd "$(dirname "$0")/../.."
names=$(grep -o '^    ("[a-z0-9_]*"' temp/e353/e353_mutants.py | sed 's/^    ("//; s/"$//')
old=$(echo "$names" | head -20)
new=$(echo "$names" | tail -n +21)
echo "new: $(echo "$new" | wc -l), old: $(echo "$old" | wc -l)"
nice python3 temp/e353/e353_mutants.py $new
nice python3 temp/e353/e353_mutants.py $old
echo "all mutants done"
