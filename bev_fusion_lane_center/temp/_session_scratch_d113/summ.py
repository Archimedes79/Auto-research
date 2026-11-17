import glob
import json
import os

R = os.environ.get("BFLC_ROOT") or "."
for d in sorted(glob.glob(os.path.join(R, "runs/e365_cache/*/*/manifest.json"))):
    m = json.load(open(d))
    n = m["numbers"]
    print(os.path.basename(os.path.dirname(os.path.dirname(d))), m["split"], "frames", n["frames"], "| cov512 nodes %.4f links %.4f | ratio384/512 %.4f | assigned %.4f valid-assigned %.4f | recall %.4f prec %.4f | reach %.4f nosrc %.4f | pairs %.0f max %d conflicts %d | e50 %.1f outsideK %.4f | nonfinite %s" % (
        n["coverage_at_K"]["512"]["nodes"], n["coverage_at_K"]["512"]["links_both_ends"], n["k_rule_ratio_384_over_512"], n["gt_nodes_assigned"], n["tokens_assigned_of_valid"], n["b_node_recall_at_r"], n["b_node_precision_at_r"],
        n["links_reachable_in_gate"], n["links_without_source_token"], n["pairs_per_frame"], n["pairs_max"], n["pair_class_conflicts"], n["cells_e50_per_frame"], n["e50_cells_outside_K"], n.get("nonfinite_values")))
