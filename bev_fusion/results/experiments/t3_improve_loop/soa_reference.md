# SOA reference numbers for the report (collected 2026-10-30; secondary sources, to be checked against the papers before the final report)

nuScenes VAL, ResNet-50 image backbone, 256 x 704 input, six cameras (+ radar where stated), all 10 classes, nuScenes mAP / NDS:

| system | modality | mAP | NDS | source |
|---|---|---|---|---|
| BEVDepth | camera | 0.351 | 0.475 | comparison table quoted in the DualBEV search result (not verified in the paper) |
| VideoBEV | camera, temporal | 0.422 | 0.535 | same table (not verified) |
| CRN | camera + radar | 0.490 | 0.560 | RCBEVDet++ paper, Table 1: [arXiv 2409.04979](https://arxiv.org/pdf/2409.04979) |
| RCBEVDet | camera + radar | 0.453 | 0.568 | same table |
| RCBEVDet++ | camera + radar | 0.519 | 0.604 | same table |
| DualBEV (our frozen image front end) | camera | not found in the search; test set V2-99: 0.552 mAP / 0.634 NDS (arXiv 2403.05402) | | [DualBEV](https://arxiv.org/pdf/2403.05402) |

Not comparable one-to-one with our numbers (stated in the report): ours is the FRONT camera only, the 60 m x 52 m front sector, three class groups (vehicle, pedestrian,
two_wheeler), mAP over matching distances 0.5 / 1 / 2 / 4 m (nuScenes uses the same distances), no attribute / velocity / NDS terms, 82 val scenes of 150, 1000 frames,
3000-step schedule (about 0.55 epoch of the 428-scene training set; published systems train about 20-24 epochs on the 700 training scenes of the official split, 850 = train + val, with 6 cameras). Qualitatively: radar-camera
fusion beats camera-only in the literature (the size of the gap on the same backbone is NOT verified here); ours: camera 0.210 -> camera + radar 0.339 mAP2 (+0.13).
