# plan - the written plan of bev_fusion

The documents that say what the camera + radar BEV system must be and why. They were in the root of `bev_fusion/` until 2026-11-01 (tidied on Mirko's word); the names did not change, so a reference such as "plan.md section 9.5" now means `plan/plan.md` section 9.5.

| file | content |
|---|---|
| `requirements.md` | what the system must satisfy, set by Mirko (camera geometry from calibration data with a general, rolled-capable, not learned projection and no virtual cameras; problem and data; method rules), each with its check and status |
| `plan.md` | the current plan (draft 3) with the paper and code basis per block, the GT generation and viewer, and the tests; its section 11 gives the stages whose ids name the experiment folders (`../experiments/README.md`). It plays the role that `PROBLEM.md` plays in the lane project; a `PROBLEM.md` for this problem would be a decision of Mirko's (protected file) and is not made |
| `radar_encoding.md` | radar grid encoding ideas (measured nuScenes radar statistics, Mirko's per-detection decoder, the symmetric polar radar image + table pull, comparison, tests); discussed separately until one is chosen |

The previous draft (`plan_draft2.md`: front camera, virtual upright camera, sigma points) left the live tree and is in `_to_delete/root_docs_2026-11-01/` (restore from git history with `git show <commit>:bev_fusion/plan_draft2.md`).
