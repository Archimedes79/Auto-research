#!/usr/bin/env bash
# one-off: render the drawing with the files that other chats added to training/ (in no box) temporarily listed as not drawn; the spec
# file is restored afterwards (those chats add their own boxes)
cd "$(dirname "$0")/../.." || exit 1
cp training/architecture.json temp/arch_build/architecture.json.keep
extra=$(python3 ../autoresearch/tools/archdraw.py check training/architecture.json 2>&1 | grep "in no box" | sed 's|ERROR: \(.*\): in no box.*|\1|' | sort -u \
        | awk '{printf "  \"%s\": \"TMP\",\n", $0}')
python3 - "$extra" <<'EOF'
import sys
p = "training/architecture.json"
s = open(p, encoding="utf-8").read()
add = sys.argv[1].rstrip(",\n")
s = s.replace(' "not_drawn": {\n', ' "not_drawn": {\n' + add + ",\n", 1) if add else s
open(p, "w", encoding="utf-8").write(s)
EOF
python3 ../autoresearch/tools/archdraw.py render training/architecture.json 2>&1 | tail -2
cp temp/arch_build/architecture.json.keep training/architecture.json
echo "temporary entries left in the spec: $(grep -c '"TMP"' training/architecture.json)"
