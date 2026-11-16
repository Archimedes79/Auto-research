import io
p = "temp/day28/day28.sh"
s = io.open(p, encoding="utf-8", newline="").read()
a = s[s.index('CMD="bash tools/env/_py14.sh jtest_eval.py all'):]
b = '''CMD="bash tools/env/_py14.sh jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cuda --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json"
if [ -n "$DAY28_DRY" ]; then say "DRY run.py --id E343.rh_gpu_dumps --gpu --no-card -- bash temp/day28/gpu_dumps.sh"; say "DRY run.py --id E343.rh_all_e324s_gpu --no-card -- $CMD"; exit 0; fi
# stage 1 (GPU slot, about 25 minutes): the four GPU dumps;  stage 2 (CPU): `all` reuses them under the same provenance (device cuda) and scores on the CPU
rp() {
  local id=$1 gpu=$2; shift 2
  local fl=""; [ "$gpu" = 1 ] && fl="--gpu"
  quiet || return 1
  sampler $D/rss_samples_$id.txt & local sp=$!
  python3 ../autoresearch/tools/run.py --id $id $fl --retries 0 --no-card --split val --timeout 720 \
    --hyp "E343 criterion 2, GPU branch: the dress rehearsal of jtest_eval.py on VAL with the pair head of E324 s (and c2's B) dumped on the GPU; diagnostic, feeds no verdict; job $id" \
    --pred "no number is predicted: the rehearsal must run to the end, reproduce the stored VAL cards of E324 s within the pre-registered tolerance and write cards with a code digest, device cuda" \
    --rule "by hand: a check of the code path, not a comparison; criterion 2 of the E343 card (GPU branch)" \
    -- "$@"
  local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  [ -s $D/rss_samples_$id.txt ] && say "peak RSS of $id: $(sort -n $D/rss_samples_$id.txt | tail -1) MB"
  return $rc
}
if [ ! -f $D/gpu_dumps.done ]; then
  wait_start
  if rp E343.rh_gpu_dumps 1 bash temp/day28/gpu_dumps.sh; then touch $D/gpu_dumps.done; say "gpu dumps done"; else touch $D/FAIL_gpu_dumps; say "gpu dumps FAILED"; exit 1; fi
fi
if rp E343.rh_all_e324s_gpu 0 $CMD; then touch $D/DAY28_DONE; say "day28 done (exit 0)"; else touch $D/FAIL_all_e324s_gpu; say "day28 FAILED"; exit 1; fi
'''
s = s.replace(a, b)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
