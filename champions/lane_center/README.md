# lane_center champions (placeholder until step S5)

Champion since D120: **E372 S4**, the end-to-end token head on the lane head B', trained together through the token loss. VAL 873 frames: topo F1 0.6682 / 0.6813 (seed 0 / 1) at the D115 decode, 16.21 ms per frame quiet (batch 1, bf16, RTX 3060), 2.323 M parameters; no TEST number. Its chain: lane head E292 a, stage 1 E373, stage 2 E372; all three weights are kept with it. Comparison networks: open. Source of these numbers today: `../../bev_fusion_lane_center/winner/e372_s4_champion/README.md`.
