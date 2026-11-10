#!/bin/bash
# for each dead candidate: who else mentions its module name in live files (outside the dead set itself, history, experiments, _to_delete, literature, runs, data)?
cd "$(dirname "$0")/../.."
while read f; do
  m=$(basename "$f" .py)
  hits=$(grep -rIlw --exclude-dir=__pycache__ --exclude-dir=history --exclude-dir=experiments --exclude-dir=_to_delete --exclude-dir=literature --exclude-dir=runs --exclude-dir=data --exclude-dir=temp --exclude-dir=plans --exclude-dir=reports -e "$m" . 2>/dev/null | sed 's#^\./##' | grep -vxF -f temp/loop23/dead.txt | grep -v "^$f$" | tr '\n' ' ')
  [ -n "$hits" ] && echo "$f <- $hits"
done < temp/loop23/dead.txt
