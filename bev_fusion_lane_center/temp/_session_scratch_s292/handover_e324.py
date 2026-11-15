p = "HANDOVER.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("2. **E324**"))
L[i] = ("2. **E324** RESULT 2026-11-14 (card section 5, ledger E324.s_s0/1): arm s = E320 recipe at 6000 pair steps. By the written rule **TIE, kept and refined (D87)**: kill not fired, dup guard breaks on seed 0 (+0.0150 vs c; seed 1 +0.0034 held). PRIMARY (attached ends) topo F1 0.6912 / 0.6948 (+0.0087 / +0.0084 over E312 snapped, +0.0206 / +0.0188 over the floor u'). "
        "Reporting: the OWN-KEYPOINT SNAPPED read-out of the same run 0.6981 / 0.7010 = +0.0156 / +0.0145 over E312 snapped with guards held: passes the E312 adoption rule = SYSTEM-OF-RECORD CANDIDATE (E312 + geometry loss + attached-end head + 6000 steps; costs as the lane lead gave them, +9.6 k params / 3.9 MMACs in training, no ms at the read-out). "
        "Dup excess of the primary on seed 0: read-out choice +0.0062, recipe +0.0017 (gone: was +0.0064), attached ends +0.0071 (CB/BC overlaps with delivered pieces, not connector-connector: arm n stays a stub). Interaction hypothesis: seed 0 no, seed 1 yes (seeds disagree). Follow-up: **E337** (card v1, critic running; step 0 CPU: attached ends with the snap's geometry, the gap +0.0069 is geometry or wiring?). "
        "**QUESTION FOR MIRKO: 'TEST E324 once?'** (adopting the candidate = a decision; any TEST use needs your recorded OK; nothing was run on TEST).")
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("ok")
