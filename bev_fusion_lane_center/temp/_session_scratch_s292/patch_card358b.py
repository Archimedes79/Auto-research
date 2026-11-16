import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 6. Summary"
assert s.count(a) == 1
gate = ("**Cheap VAL gate (CPU, 2026-11-16 11:06, diagnostic by hand, IN-SAMPLE on VAL, never a claim; `temp/e358/gate.sh`, logs `temp/e358/gate_*.log`, caches `temp/jwire/val_records_v1_s<S>_pair_e324_s_s<S>_ends_att.pkl`):** attached-end records of the stored dumps: 873 frames, 16969 / 17331 ends of which 12773 / 12934 carry an oracle label; **the rule's agreement with the oracle label overall 0.864 / 0.854, on the real choices 0.764 / 0.748 (7388 / 7504 ends = 57.8 % / 58.0 % of the labelled ends)**; the attached-end error of the matched ends mean 1.31 / 1.31, median 1.09 / 1.13, p90 2.73 / 2.62 m; share of the labelled ends whose nearest piece end lies within 0.5 m 0.478 / 0.464 (unperturbed VAL). **5-fold scene-wise cross-validation (E325's tool): real choices head 0.773 / 0.768 against the rule 0.764 / 0.748 (+0.009 / +0.020), all labelled ends 0.869 / 0.866 against 0.864 / 0.854; the distance-only head equals the rule (0.767 / 0.749 on the real choices).** Gate rule as written (NO-GO only if the head is more than 0.02 below the rule on the real choices on either seed, or a seed has fewer than 500 real choices): **GO on both seeds.** (The first draft's gate asked for +0.01 on both seeds: seed 0 would have missed it by 0.001, which is why it was changed on the critic's finding 4 before any number existed.)" + nl + nl)
s = s.replace(a, gate + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
