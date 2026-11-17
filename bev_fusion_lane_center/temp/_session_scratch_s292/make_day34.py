"""Builds temp/day34/day34.sh from temp/day29/day29.sh: the same functions and markers (temp/day29/*.done), but the post stages of the four runs
run as TWO parallel workers (seed 0: n1_s0, cp_s0, n1z_s0; seed 1: n1_s1, cp_s1, n1z_s1) and the rule replay follows when both are through.
The dumps stay one at a time (flock /tmp/jends_dump.lock inside the run.py job), the cards run on the CPU. Waits for temp/day29/train_s1.done first."""
import io

src = io.open("temp/day29/day29.sh", encoding="utf-8", newline="").read()
head, sep, _ = src.partition("for r in n1_s0 cp_s0 n1_s1 cp_s1; do")
assert sep, "marker of the post loop not found"
head = head.replace("exec 9>/tmp/day29${DAY29_DRY:+_dry}.lock", "exec 9>/tmp/day34${DAY29_DRY:+_dry}.lock")
head = head.replace("flock -n 9 || { echo \"day29 already runs $(date)\" >> $L; exit 0; }", "flock -n 9 || { echo \"day34 already runs $(date)\" >> $L; exit 0; }")
head = head.replace("mkdir -p $D\n", "mkdir -p $D\nexport DAY29_SKIP_WAIT=1\nwhile [ ! -f $D/train_s1.done ] && [ -z \"$DAY29_DRY\" ]; do sleep 30; done\n", 1)
assert "train_s1.done" in head and "/tmp/day34" in head and "day34 already" in head
head = head.replace("# Day chain 29", "# Day chain 34 = day29 with the post stages in two parallel workers (D107: finish by 07:30); the functions and markers are day29's. Was: Day chain 29", 1)
tail = '''workerS() { # <seed>: the post stages of the runs of one seed, then its step-0 row
  local S=$1 r
  for r in n1_s$S cp_s$S; do stage post_$r "post $r" || return 1; done
  stage post_n1z_s$S "post_z $S"
}
workerS 0 & pa=$!
workerS 1 & pb=$!
wait $pa || exit 1
wait $pb || exit 1

rule() {
  rp E353.rule_c 0 0 bash tools/env/_py14.sh jrule_removal.py --arm e353_n1_s --control e324_s_s --readouts snap primary attach
  rp E353.rule_cp 0 0 bash tools/env/_py14.sh jrule_removal.py --arm e353_n1_s --control e353_cp_s --readouts snap primary attach
}
stage rule rule
touch $D/DAY29_DONE
say "done (day34)"
'''
import os
os.makedirs("temp/day34", exist_ok=True)
io.open("temp/day34/day34.sh", "w", encoding="utf-8", newline="").write(head + tail)
print("ok", len(head + tail))
