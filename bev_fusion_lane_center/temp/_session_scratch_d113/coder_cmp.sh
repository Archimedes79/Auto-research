cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
import json, re, sys
L = "temp/_session_scratch_d113/"
bad = 0
for seed in (0, 1):
    j = json.load(open(L + f"coder_potential_{seed}.json"))
    a = open(L + f"coder_ref0_{seed}.log").read()
    b = open(L + f"coder_ref0b_{seed}.log").read()
    exp = []
    exp.append("GT nodes of frames with a past: %d; found by a delivered token (e >= 0.5) %.4f; missed %.4f" % (j["gt_nodes"], j["found"], j["missed"]))
    exp.append("a weak token (0.05 < p < 0.5) is near %.4f, no token at all %.4f; a delivered token of t-1 (warped) is near %.4f; weak AND past support %.4f" % (j["missed_weak_near"], j["missed_none"], j["missed_delivered_past_near"], j["missed_weak_and_past"]))
    w, d = j["weak"], j["delivered"]
    exp.append("weak tokens (valid, p < 0.5): true (near a GT node): %d, supported by the past %.4f | false: %d, supported %.4f" % (w["true"], w["true_supported"], w["false"], w["false_supported"]))
    exp.append("delivered tokens (p >= 0.5): true (near a GT node): %d, supported by the past %.4f | false: %d, supported %.4f" % (d["true"], d["true_supported"], d["false"], d["false_supported"]))
    for r in j["by_bin"]:
        exp.append("  %.2f - %.2f | %7d | %.3f | %.3f | %.3f" % (r["lo"], r["hi"], r["n"], r["true_share"], r["supported_if_true"], r["supported_if_false"]))
    for line in exp:
        ok = line in a
        bad += not ok
        print(seed, "ok " if ok else "BAD", line[:110])
    exp = []
    exp.append("VAL frames with a past: GT nodes %d missed %d (%.4f)" % (j["gt_nodes"], round(j["missed"] * j["gt_nodes"]), j["missed"]))
    for k, nm in (("novel", "novel "), ("dup", "dup   "), ("false", "false ")):
        r = j["roles"][k]
        exp.append("  %-6s %7d | %.3f | %.3f | %.3f" % (k, r["n"], r["pose"], r["identity"], r["random"]))
    exp.append("DUPLICATES of an already found GT node: %.3f" % j["duplicates_of_the_true_weak"])
    rc = j["recall_ceiling"]
    exp.append("as node recall: +%.4f; of it on topology nodes (stop/exit/split/merge/start/end): +%.4f; start/end only: +%.4f" % (rc["all"], rc["topology"], rc["start_end"]))
    exp.append("topology nodes: %.4f of all GT nodes; of the missed %.4f; start/end missed %.4f of all missed" % (j["topology_share_of_gt"], j["topology_share_of_missed"], j["start_end_share_of_missed"]))
    for line in exp:
        ok = line in b
        bad += not ok
        print(seed, "ok " if ok else "BAD", line[:110])
print("mismatches:", bad)
EOF
