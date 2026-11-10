# jobs/training - job files of kind `train` (suite: `autoresearch/trainsuite`, class `Training`)

```
cd autoresearch                                                  # from the workspace root, inside wsl
AR_PROJECT=../bev_fusion python3 -m trainsuite Training.check ../bev_fusion/records/jobs/training/example_t23_lane_bce.json
```

| file | what |
|---|---|
| `example_t23_lane_bce.json` | the fine-tune of t23 (tolerance-aware lane-line BCE) as a job: config, preset, seed, steps. Checked only (`no_card`: a diagnostic, it feeds no verdict); a real run gets its card |

Fields: `id`, `kind`, `config`, `preset`, `set`, `card`, `no_card`, `after`, `priority`, `kill`, `note` (reference: `../../../../autoresearch/trainsuite/README.md`, section 0).
