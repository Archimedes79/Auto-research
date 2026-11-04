# results/build - the result report of the BEV system

Builds the report page (`report.html`, images embedded, videos as files next to it), its proofreading text, the evaluation tables and the plain-text mail from the **stored** evaluation files of the runs. Promoted on 2026-11-03 from the scratch of the t16 report (v6, `temp/t16/build_big_report.py` and its helpers); the same inputs give the same page (proof below). The numbers are read, never typed here: the prose lives in text files, the rows and runs in `report_runs.json`.

## Build the v7 report (one command, from the repository root, WSL)

```
nice -n 15 ~/miniconda3/envs/gcar/bin/python bev_fusion/results/build/build_report.py
```

This writes `bev_fusion/results/BEV_result_report_v7/` (`report.html`, `artifact_page.html`, `report.txt`, `tables/`, `videos/<run>/`, `README.txt`, `mail.txt`, `check_numbers.txt`) and prints the first line of the numbers check. A missing input stops the build with a message (`report build FAILED: ...`); nothing is skipped silently. It needs, besides the evaluation of `t20_base_s0` (`evaluate.py ... --all`, with `timing`), the paired comparison of the new run against its control (CPU, a minute):

```
cd bev_fusion && python evaluation/bev_suite.py compare runs/t16_full_s0 runs/t20_base_s0 --curves --fusion --velocity --n 1000 --boot 1000 > experiments/t20_lane_grid_bundle/compare_vs_t16_full_s0.txt
```

(`compare` bootstraps only the headline numbers of the tasks (`BevSegMetric.headline` in `evaluation/bev_task.py`); the other rows of the bundle table say `n/a (no paired test for this metric)` in the paired column and keep their unpaired difference. A compare output made without `--velocity` or `--fusion` is refused.) Then write `chapters/t20.md` (the chapter), `chapters/mail_t20.txt` (the mail) and the `summary` block of `chapters/text_v7.txt`.

**Test build before the new run exists:** `build_report.py --standin t18_fix --compare results/build/inputs/compare_t18_vs_base.txt` lets `t18_fix` play the new run, writes into `..._v7_test/` and puts a TEST banner on the page.

## Add a run

One entry in `runs` of `report_runs.json` (`id` = folder under `runs/`, `label`, `role`: `baseline` (exactly one: the run the lane, picture and summary parts read) / `control` / `arm`, `long` = the heading of its videos; `optional: true` skips it while it has no evaluation; `eval`, `videos` only when the folders differ from `runs/<id>/eval` and its `showcase` / `suite` videos), one `cost_rows` entry (the ledger id and a label), and the run in `mail.runs` if the mail table should show it. A run's train table (`eval/table_train.txt` or `eval/by_n/table_train_n1000.txt`, made by `bev_suite.py RUN generalisation`) is optional: without it the run has no column in the train-against-validation table and `-` as the gap of the weak-spot table; every other file of an enabled section is required. To change what the bundle chapter shows, edit `bundle.rows` (`name`, `key` of the evaluation table, `lower`, `weak`, `new_only`, `timing`).

## The bundle chapter ("the t20 bundle")

Section `bundle` of `sections` (the slot sits after the summary). Its text is `bundle.chapter` (`chapters/t20.md`: markdown with headings, paragraphs, `- ` lists and pipe tables); a line `@@TABLE@@` puts the numbers table there, a line `@@WEAK@@` the weak-spot table; without them both come first. All numbers come from `bundle_numbers.py`, which holds the rules:

| table | rule |
|---|---|
| numbers (VAL, 1000 frames) | per row: the new run, the control (evaluation tables; the ms row from the idle-GPU `timing_modules.json`), the difference (new - control; green / red beyond `bundle.band` by the direction of the row), the paired bootstrap `b - a [95 % interval]` from the compare output (b = new, a = control), the reading: interval excludes 0 -> `better` / `worse` by the direction, else `not separated`, no paired row -> `n/a` (the paired cell says `no paired test for this metric`, or `the control has no such value`; the difference column is then the unpaired one) |
| weak spots | the rows with `weak: true`, lowest value of the new run first, flagged `weak` below `bundle.weak_below`; with the new run's train - validation gap (its train table) and the paired value; it updates itself with the numbers |
| summary | the same rows, compact (`summary.table: "bundle"`) |

A run lacking a key, a compare output about other runs (its header names a and b), values in the compare output that disagree with the tables, a compare output without a whole group (`vel.`, `detf.`, forgotten flag), a missing or non-idle timing all stop the build. A row of a head the control never had says `new_only`.

## Files

| file | what |
|---|---|
| `build_report.py` | the builder: config, runs, sections, facts, page assembly, out folder, numbers check; `--standin`, `--out`, `--no-mail`, `--no-check` |
| `bundle_numbers.py` | the bundle table, the reading rule, the weak-spot rule, the compact summary table (pure functions on the rows) |
| `report_lib.py` | readers of the stored files (table, metrics, frames, compare output), the text bank, `@@name@@` filling, markdown / html pieces, charts |
| `build_mail.py` | the plain-text mail from its narrative file (`@@MAIL_TABLE@@`, `@@MAIL_COST@@`, `@@MAIL_SOA@@` and every fact) |
| `report_runs.json` | the v7 config: runs and roles, sections, bundle rows, inputs, ladder, cost rows, mail, sources of the numbers check |
| `report_runs_v6.json` | the v6 config: rebuilds the v6 report (`test_report_build.py --real`) |
| `page.css` | the style of the page |
| `inputs/` | the stored inputs of the sections 2, 4, 5 and 6 (digests, re-scorings, compare outputs, lane picks; table below) |
| `chapters/text_v6.txt`, `chapters/text_v7.txt` | text banks: blocks `@@@ name` of html prose with `@@fact@@` markers (v7 overrides `lead`, `summary`, `folder_readme`) |
| `chapters/t20.md`, `chapters/mail_t20.txt` | the chapter and the mail of the t20 bundle (written by hand once the numbers exist; stubs until then) |
| `chapters/mail_v6.txt` | the v6 mail |

Facts a text may ask for (`@@name@@`, computed from the files when first asked): `head`, `frames`, `done`, `new_label`, `ctl_label`, `key_table`, `lane_n`, `lane_median`, `lane_lo`, `lane_hi`, `zc_p`, `zc_r`, `zc_f1`, `lane_plain`, `vel_field`, `vel_zero`, `vel_zero3`, `gap6`, `nvid`, `spread`, `ms_<run id>`, `architecture_svg`, `soa_sentence` and the numbers of the state-of-the-art re-scorings (`front_dual`, `front_ours`, `front_diff`, `soa_parts`, `maps_table`, `carpark_soa2` ...). An unknown name stops the build.

## Checks

- `tests/test_report_build.py` (in `tests/run_gates.sh`, 8 s, CPU): a fixture report, every number in a source (`autoresearch/tools/check_numbers.py --strict`), every cell in place, the missing-input cases, 20 mutants.
- The numbers check at the end of a build (`check_numbers.txt`) lists the decimal numbers of the page that no source holds: the prose of the text banks (numbers typed by hand in the narrative), the differences the page computes. Advisory.
- **v6 proof (2026-11-03):** the v6 config builds `report.html`, `artifact_page.html`, `mail.txt`, `tables/` and `videos/` byte for byte equal to a rebuild by the original scratch code from the same inputs. Against the stored `results/BEV_result_report/report.html` (until 2026-11-04 also `results/t16/report_big.html`, byte-identical; that folder was deleted as a duplicate) the only differences are the commit hash of the lead paragraph, the architecture drawing (redrawn since) and the folder pointer in the last list (it named `results/t16/`, now it names no folder because the v7 build shares the text); the stored `mail.txt` has a BOM and CRLF line ends (re-saved on Windows), the stored `README.txt` was edited by hand afterwards. `python tests/test_report_build.py --real` repeats the comparison.

Inputs: nothing of the builder or its configs reads from a `temp/` folder (the gate asserts it, and that every file a config names exists). The inputs of the v6 report were copied byte for byte from the scratch of the t16 report into `inputs/` (each under 60 KB); they are results of scratch scripts of that report (in its scratch tag), not made by this builder:

| file in `inputs/` | what | how it was made / rebuilt |
|---|---|---|
| `history_digest.md`, `soa_digest.md` | the digests of the experiment cards and of the state-of-the-art numbers (section 2, 6.1 / 6.2 / 6.4) | by a read-only agent pass over the cards and the literature notes (2026-11-02 / 03), spot-checked; written by hand again when a card or a paper changes |
| `soa_maps_v8_t16_full_s0.md`, `soa_maps_v8_t17_data.md`, `front_v8_n1000.md` | the re-scoring of GaussianCaR's cached map logits against our baseline / t17 (384 frames) and of DualBEV in the front sector (1000 frames) | scratch scripts `soa_maps_v8.py`, `soa_front_v8.py` (scratch tag of the t16 report); their pickles and the 1.6 MB `soa_front_v8_inputs.json` stay out; a new run needs the scripts promoted first |
| `compare_t17.txt`, `compare_t18.txt`, `compare_t18_vs_base.txt` | the paired bootstrap outputs of section 4 | `bev_suite.py compare runs/<a> runs/<b> --curves --fusion --velocity --n 1000 --boot 1000` (the header line of each file names a and b; the scratch scripts `compare_t17.sh`, `compare_t18.sh` hold the commands) |
| `lane_frames_t16_full_s0.txt` | the worst / median / best lane-divider frames of the baseline (gallery of section 5) | scratch `pick_lane_frames.py` |

The lane reading of section 5 (four frames, in words) is the text block `lane_reading` of `chapters/text_v6.txt`. `temp/t16/report_static.md` is a document, no input of the builder. Not promoted: `build_report.py` of the t16 scratch (the earlier 9 MB page, superseded) and `build_tables.py` (markdown tables for cards).
