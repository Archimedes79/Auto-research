cd /mnt/c/Auto-research/bev_fusion
AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/ledger.py note t10.dual_s1 \
  --result "3000 steps done 02:50 (done flag); attempt 1 (snapshot 2f25f4b) OOM-killed by RAM at step 760 (00:09, side jobs of several chats), resumed from ckpt 500 on snapshot 77aed27 (default-off additions only: t11 warm start / hurdle, the backbone chat's camera-checkpoint / check_grids, all inactive for this config); cold VAL 1000 eval = the BEV system chat's run.py item t10.dual_s1.eval" \
  --decision "pending its eval: card rule E1 of t10_dual_features (board base proposal; promotion stays Mirko's)"
AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/ledger.py note t8.ctl_s1 \
  --result "resumed 00:12 from ckpt 2500 on today's snapshot (base seed 2, CODE DIFFERS from 96a02be for the last 500 steps); its run.py launcher died when the lead stopped its queue script, training finished through run_ar.sh (done flag); eval = run.py item t8.ctl_s1.eval (another chat)" \
  --decision "measurement for the board's base band"
