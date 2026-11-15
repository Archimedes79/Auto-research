"""mutation check of qgraph_eval: apply one text mutation to the module, run the named tests against it, report whether they FAIL (= the rule is pinned).
The mutant source goes to mutants/<name>.txt (not a .py: it must not shadow qgraph_eval on the project's flat module path) and is loaded under the name qgraph_eval."""
import importlib.machinery, importlib.util, os, sys
import bflc_paths as BP
MUT = {
    "no_link_pieces": ("    for l in L[cross]:   ", "    for l in L[cross][:0]:   "),
    "tie_zero": ("TIE = 0.3 ", "TIE = 0.0 "),
    "link_threshold_ignored": ("ok = (p >= thr_link) & keep[src]", "ok = (p >= -1) & keep[src]"),
    "dropped_nodes_keep_links": ("& keep[src] & keep[dst] & (src != dst)", "& (src != dst)"),
    "edges_from_start": ("starts.get(n[-1], [])", "starts.get(n[0], [])"),
    "section_ignored": ("cross = sec[src[L]] != sec[dst[L]] if len(L) else np.zeros(0, bool)", "cross = np.zeros(len(L), bool)"),
    "dust_not_far": ("true_dust = True                                                               # farther", "true_dust = False                                                              # farther"),
}
name, words = sys.argv[1], sys.argv[2:]
src = open(f"{BP.ROOT}/evaluation/qgraph_eval.py", encoding="utf-8").read()
a, b = MUT[name]
assert src.count(a) == 1, (name, src.count(a))
path = f"{BP.ROOT}/temp/e273_scorer/mutants/{name}.txt"
open(path, "w", encoding="utf-8").write(src.replace(a, b))
spec = importlib.util.spec_from_file_location("qgraph_eval", path, loader=importlib.machinery.SourceFileLoader("qgraph_eval", path))
mod = importlib.util.module_from_spec(spec)
sys.modules["qgraph_eval"] = mod
spec.loader.exec_module(mod)
sys.path.insert(0, f"{BP.ROOT}/evaluation/tests")
import test_qgraph_eval as T
assert T.QE is mod
rc = T.main(words)
print(f"MUTANT {name}: tests {'FAILED (killed)' if rc else 'PASSED (survived!)'}")
