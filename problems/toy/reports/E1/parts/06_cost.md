Table: Inference. One frame at batch 1 on the CPU with two threads: the forward pass and the threshold, no loading; measured one network after the other (1-minute load {{load}}).

| run | ms per frame (median) | ms (p90) | parameters | GMACs per frame |
|---|---|---|---|---|
| 4 channels, seed 0 | {{ms_c0}} | {{p90_c0}} | {{par_c}} | {{mac_c}} |
| 4 channels, seed 1 | {{ms_c1}} | {{p90_c1}} | {{par_c}} | {{mac_c}} |
| 12 channels, seed 0 | {{ms_a0}} | {{p90_a0}} | {{par_a}} | {{mac_a}} |
| 12 channels, seed 1 | {{ms_a1}} | {{p90_a1}} | {{par_a}} | {{mac_a}} |

The operation count covers convolutions only; element-wise operations are not counted.

Table: The experiment. Wall time of each run (training and evaluation, from the run record) and the speed of the training steps (from the training log). No GPU was used.

| run | wall seconds | training steps per second | waiting for data |
|---|---|---|---|
| 4 channels, seed 0 | {{wall_c0}} | {{sps_c0}} | - |
| 4 channels, seed 1 | {{wall_c1}} | {{sps_c1}} | - |
| 12 channels, seed 0 | {{wall_a0}} | {{sps_a0}} | - |
| 12 channels, seed 1 | {{wall_a1}} | {{sps_a1}} | - |

All four runs together: {{wall_total}} seconds of run time. The card's budget was zero GPU hours. The share of waiting for data was not measured: the trainer sets no timing hook.
