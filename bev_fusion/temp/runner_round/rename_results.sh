#!/bin/bash
# scratch: Runner.results -> Runner.list_results (verb first), in the files of runsuite only
cd /mnt/c/Auto-research/autoresearch/runsuite || exit 1
sed -i 's/"results", "tap", "time", "probe"/"list_results", "tap", "time", "probe"/; s/def results(self, network)/def list_results(self, network)/; s/self\._p()\.results(network)/self._p().list_results(network)/' api.py
sed -i 's/def results(self, network: str)/def list_results(self, network: str)/' predict.py
sed -i 's/{"predict", "plan", "results", "tap", "time", "probe"/{"predict", "plan", "list_results", "tap", "time", "probe"/; s/r\.results(net)/r.list_results(net)/g; s/cli("results", net/cli("list-results", net/' selftest.py
sed -i 's/r\.results("runs\/t23_C_s1")/r.list_results("runs\/t23_C_s1")/; s/`plan` (the dry run), `results`/`plan` (the dry run), `list_results`/' __init__.py
sed -i 's/`plan` (the dry run), `results`, `tap`/`plan` (the dry run), `list_results`, `tap`/; s/`predict` \/ `plan` \/ `results` \/ `tap`/`predict` \/ `plan` \/ `list_results` \/ `tap`/g' README.md
sed -i 's/`predict`, `plan`, `results`, `tap`, `time`, `probe`/`predict`, `plan`, `list_results`, `tap`, `time`, `probe`/' __main__.py
sed -i 's/`predict` \/ `plan` \/ `results` \/ `tap` \/ `time` \/ `probe`/`predict` \/ `plan` \/ `list_results` \/ `tap` \/ `time` \/ `probe`/' ../README.md
grep -n "results" api.py predict.py __init__.py __main__.py README.md | grep -v "list_results" | grep -iE "def results|\.results\(|\"results\"|\`results\`" | head
echo "--- list_results occurrences:"; grep -c "list_results" api.py predict.py selftest.py __init__.py README.md __main__.py ../README.md
