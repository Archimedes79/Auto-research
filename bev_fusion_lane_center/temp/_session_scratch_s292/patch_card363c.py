import io
p = "experiments/E363_e2e-stack-on-faithful-graphs/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
add = io.open("temp/_session_scratch_s292/e363_summary.md", encoding="utf-8").read().replace("\n", nl)
a = "## 6. Summary" + nl + "(empty until the runs exist)"
assert s.count(a) == 1
s = s.replace(a, "## 6. Summary" + nl + add.rstrip() + nl)
b = "## 4. Referee" + nl + "(empty: the first fresh ar-critic pass, mode cards, is the next step)"
assert s.count(b) == 1, s.count(b)
s = s.replace(b, "## 4. Referee" + nl + "**First pass 2026-11-16 14:00 (fresh ar-critic, mode cards): REVISE, G2 no** (blocking: arm a without the Matrix-NMS layer in front of `Attach`, the symmetry stop rule dropped; should fix: array stop rules, the exact fixture on both seeds, the head's hit window, the faithful arm t, one fine-tune draw, the recorded deviations; confirmed: the surgery, the pair model's use of the nodes, the fixture bit for bit, the scorer command, the numbers). **Second pass 14:15 (the same critic, verification): G2 YES** after the layer-faithful dump with its node check, the kept symmetry rule and the witness condition, the array stop rules and the rest; remarks taken (stale text of section 2 and the verdict docstring fixed; `start_faithful.sh` passed `--seed 0` to the seed 1 data-preparation jobs: only the ledger seed column is wrong for those four rows; the critic corrected its own cause for the shift of a's attached ends: the layer explains about 0.02 m, most of the 0.17 m comes from a's node head).")
s = s.replace("**Status: card v2; the dumps of arm c", "**RESULT 2026-11-16 14:45: TIE, kept and refined (section 6): the stack aw 0.6820 / 0.6854.** **Status (at the time of writing the card): card v2; the dumps of arm c", 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
