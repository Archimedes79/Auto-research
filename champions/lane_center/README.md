# lane_center champions

Champion: **`E396.champ`** = E372 S4 (D120), the end-to-end token head on the lane head B', trained together through the token loss; registered
by the research system (`research register`). Its facts are in `E396.champ/network.json`. VAL 873 frames: topo F1 0.668172 / 0.681281
(seed 0 / 1) at the D115 decode, the numbers of tag `v1` exactly; 16.21 ms per frame (the timing of `v1`: quiet GPU, batch 1, bf16, RTX 3060);
2.323 M parameters; no TEST number. Its chain: lane head E292 a, stage 1 E373, stage 2 E372; the three weights are kept in the runs root. Comparison networks: open.

Code: `../../problems/lane_center/`.
