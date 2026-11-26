Table: VAL split, {{n_val}} frames in {{n_scenes}} scenes. Pooled pixel counts; one row per run.

| arm | seed | IoU | precision | recall |
|---|---|---|---|---|
| 4 channels (control) | 0 | {{iou_c0}} | {{pre_c0}} | {{rec_c0}} |
| 4 channels (control) | 1 | {{iou_c1}} | {{pre_c1}} | {{rec_c1}} |
| 12 channels | 0 | {{iou_a0}} | {{pre_a0}} | {{rec_a0}} |
| 12 channels | 1 | {{iou_a1}} | {{pre_a1}} | {{rec_a1}} |

Table: 12 channels minus 4 channels, IoU, paired bootstrap over the scenes.

| seed | difference | interval low | interval high |
|---|---|---|---|
| 0 | {{d0}} | {{d0lo}} | {{d0hi}} |
| 1 | {{d1}} | {{d1lo}} | {{d1hi}} |

A control whose precision is high and whose recall is near zero predicts almost nothing: its few marked pixels are right.
