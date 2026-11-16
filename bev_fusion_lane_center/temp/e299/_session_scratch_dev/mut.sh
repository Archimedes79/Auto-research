#!/bin/bash
# real-code mutations of the scratch jstate.py / lane_gt_data.py: the suite must fail on each
cd /mnt/c/Auto-research/bev_fusion_lane_center
D=temp/e299/_session_scratch_dev
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { # name, file, sed-expr
  rm -rf $D/_mut && mkdir -p $D/_mut && cp -r $D/training $D/_mut/
  sed -i "$3" $D/_mut/training/$2 
  if cmp -s $D/training/$2 $D/_mut/training/$2; then echo "MUTATION $1: sed did not change the file"; return; fi
  out=$(nice -n 10 bash tools/env/_py14.sh $D/_mut/training/grid/test_jstate.py 2>&1 | tail -3)
  if echo "$out" | grep -q "test_jstate ok"; then echo "MUTATION $1 SURVIVED"; else echo "MUTATION $1 killed: $(echo "$out" | tail -1 | cut -c1-150)"; fi
}
run warp-direction grid/jstate.py 's/Rt = R.transpose(1, 2)/Rt = R/'
run no-detach grid/jstate.py 's/return {k: v.detach() for k, v in st.items()}/return st/'
run gate-nonzero grid/jstate.py 's/self.gate = nn.Parameter(torch.zeros(1))/self.gate = nn.Parameter(torch.full((1,), 0.1))/'
run state-ignored grid/jstate.py 's/off = (w\[..., None\] \* (state\["pos"\]/off = (0 * w[..., None] * (state["pos"]/; s/(w @ sf).to(t.dtype)/(0 * (w @ sf)).to(t.dtype)/; s/w.sum(-1, keepdim=True).to(t.dtype)\]/0 * w.sum(-1, keepdim=True).to(t.dtype)]/'
run next-row data/lane_gt_data.py 's/prow = int(pose\["prev"\]\[row\])/prow = int(pose["prev"][row]) + 2/'
run no-has-mask grid/jstate.py 's/return torch.where(state\["has"\]\[:, None, None\], t + (self.gate \* upd).to(t.dtype), t)/return t + (self.gate * upd).to(t.dtype)/'
rm -rf $D/_mut
