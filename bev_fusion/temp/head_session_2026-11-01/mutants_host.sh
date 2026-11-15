#!/usr/bin/env bash
# mutants of the host-side code (build_targets numpy fields, _stack_many): every one must fail tests/test_host_targets.py (bit-exact), CPU, seconds
S=/mnt/c/Auto-research/bev_fusion
M=$S/temp/head_session_2026-11-01/mut
PY=~/miniconda3/envs/gcar/bin/python
rm -rf "$M"; mkdir -p "$M/bev_fusion/tests/support"
cp -r $S/training "$M/bev_fusion/training"
cp $S/tests/test_host_targets.py "$M/bev_fusion/tests/"; cp $S/tests/support/synthetic_world.py "$M/bev_fusion/tests/support/"
ln -s $S/data "$M/bev_fusion/data"
ln -s /mnt/c/Auto-research/autogt "$M/autogt"
n=0; killed=0
mut() {  # mut <name> <file> <sed expression>
  n=$((n+1)); f="$M/bev_fusion/training/$2"; cp "$S/training/$2" "$f"
  sed -i "$3" "$f"
  if cmp -s "$S/training/$2" "$f"; then echo "MUTANT $1: sed changed nothing (bad mutant)"; return; fi
  if (cd "$M/bev_fusion" && OMP_NUM_THREADS=2 nice -n 10 $PY tests/test_host_targets.py > "$M/out.log" 2>&1); then echo "MUTANT $1: SURVIVED"
  else killed=$((killed+1)); echo "MUTANT $1: killed ($(grep -E 'Error' "$M/out.log" | tail -1 | cut -c1-80))"; fi
  cp "$S/training/$2" "$f"
}
mut np_exp        object_head.py 's/peaks = torch.exp(torch.from_numpy(np.concatenate(args))).numpy()/peaks = np.exp(np.concatenate(args))/'
mut cap_off_by_1  object_head.py 's/\[0\]\[:max_obj\]/[0][:max_obj - 1]/'
mut window_short  object_head.py 's/min(X, cx_ + r_ + 1)/min(X, cx_ + r_)/'
mut dir_flipped   object_head.py 's/% (2 \* math.pi) < math.pi), 0.0)/% (2 * math.pi) > math.pi), 0.0)/'
mut overwrite_max object_head.py 's/np.maximum(win, peaks\[c:c + m\].reshape(x1 - x0, y1 - y0), out=win)/win[...] = peaks[c:c + m].reshape(x1 - x0, y1 - y0)/'
mut off_sign      object_head.py 's/np.stack(\[fx - ix, fy - iy\], 1)/np.stack([ix - fx, fy - iy], 1)/'
mut rad_for_box   object_head.py 's/rad\[b, :n, 0\] = np.where(isb, 0.0, np.log(radius))/rad[b, :n, 0] = np.where(isb, 0.0, np.log(radius) + 1e-6)/'
mut stack_offset  losses.py 's/        c += p.shape\[1\]/        c += 1/'
mut stack_f64     losses.py 's/a = np.stack(\[np.asarray(g\[k\], dtype=np.float32) for g in gts\])/a = np.stack([np.asarray(g[k], dtype=np.float64) for g in gts])/'
mut mask_ignores_loss_w object_head.py 's/mask\[b, :n\] = \[bx.get("loss_w", 1.0) for bx in bb\]/mask[b, :n] = 1.0/'
mut known_always_ones   object_head.py 's/known\[b, :n\] = \[1.0 if bx.get("vel_known", True) else 0.0 for bx in bb\]/known[b, :n] = 1.0/'
mut group_before_filter object_head.py 's/        g = np.array(\[gidx\[bx\["group"\]\] for bx in bb\])  /        g = np.array([gidx[bx["group"]] for bx in boxes])[sel]  /'
mut stack_order  losses.py 's/    keys = list(dict.fromkeys(keys))/    keys = sorted(set(keys), reverse=True)/'
echo "killed $killed of $n"
rm "$M/bev_fusion/data" "$M/autogt"      # unlink the symlinks first, then the copy
rm -rf "$M"
