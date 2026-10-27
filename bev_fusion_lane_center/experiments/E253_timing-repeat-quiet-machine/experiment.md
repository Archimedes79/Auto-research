# E253 Timing repeat of the chain on a quiet machine (E243 session 2)

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E253 / H20.S1 / E243 repeated; the session of loop 19 was invalid (machine 2 - 3 ms above the stored values, spread > 1.5 ms in 5 of 6 configurations) |
| kind | variant (timing only, deterministic checkpoints) |
| metric, direction, band | ms_per_frame, lower, 1.0 ms |
| prediction | j1 (a) minus champion without recursion (c): -0.3 ms (TIE expected) |
| arms | a j1 192/96 fp 0.3, c f1 288/144 fp 0.3, b j1eq, d j1rc, e rc1 288/144, f f1 with time recursion; six interleaved rounds, fresh process per replicate, 200 timed frames |
| rule | KEEP if a is faster than c by >= 1.0 ms with all replicate pairs on the same side; CUT if slower by > 1.0 ms; else TIE; asm_ms guard 1.0 |
| budget, kill | 0.7 GPU h exclusive (about 45 min wall); void a replicate when GPU is busy or the 1-minute load average is above 3 (new guard) |
| evaluate | `temp/loop20/analyse_e253.py` (medians per configuration, spread, reproduction of the stored values) |

## 1. Card
| | |
|---|---|
| lens | system review (measurement protocol) |
| group | A1 speed |
| sources | E243 (invalid session: order only j1 < f1 < j1eq < rc1 < j1rc < f1 + recursion), loop-18 timings, D4 (cost per variant) |
| status | proposed |

- **Statement**: with numba assembly and fork pruning the fast variant j1 (192 / 96) is not faster than the champion without time recursion (288 / 144); the recall variants cost +2 to +3 ms; the time recursion costs about 6 ms.
- **Mechanism**: the builder is a small part of the chain; the assembly (CPU, numba) and the point count dominate, and the 192 / 96 variant yields the same number of vertices per frame.
- **Prediction** {value -0.3 ms, tolerance 0.5 ms, class "timing difference of two chains with the same vertex count"}: a - c = -0.3 ms (decide.py sign: lower is better, so the improvement is +0.3 ms and the clause is read as a - c <= -1.0 ms); j1rc - j1eq = +2.5 ms (tolerance 1.0); f1 time recursion minus f1 = +6 ms (tolerance 1.5); each of a, c, f reproduces its stored value within 1.0 ms. The three bundled claims are scored separately in the summary.
- **Decision rule**: as in section 0; the session is valid only when (1) at least five valid replicates exist in each of the six configurations (voids are repeated in at most two repair rounds), (2) the replicate SD is <= 0.6 ms in at least five of six, (3) a, c and f reproduce their stored values within 1.0 ms (rc1 has no stored value and is judged on SD only); otherwise INVALID and the clause is not read. A TIE again means "j1 is not a speed variant".
- **Test design**: `temp/loop20/run_e253.sh` (E243 driver plus guards: GPU idle, WSL 1-minute load average <= 3, Windows total CPU <= 20 % through powershell.exe; load, Windows CPU, GPU clock and temperature are recorded per replicate in `m_<arm>_r<k>.txt`), run LAST in the loop with nothing else scheduled; other chats were asked to stay off the machine, which is a request and not a lease (PLAN N1); analysis `temp/loop20/analyse_e253.py` (replicate SD, session validity, stored-value reproduction).
- **Cost / kill**: 45 min wall, exclusive; stop at 1 h.
- **Inputs**: ready.
- **Referee** (2026-10-26, ar-referee): REVISE (minor, then ACCEPT; the question is half of the objective and E243's invalid session is exactly why it is repeated; the decision rule can come out against the idea). Revise: (1) GUARD: `/proc/loadavg` inside WSL does not see Windows-side load (other chats' Windows processes, browser, the pending tools chat), is a 1-minute average that lags, and `nvidia-smi --query-compute-apps` often lists nothing under WSL; record per replicate the Windows CPU utilisation (e.g. `powershell Get-Counter '\Processor(_Total)\% Processor Time'` called from WSL) and CPU clock, and void above a stated threshold; also log GPU clock / temperature (N7) so a drift over the rounds is visible. (2) SPREAD CRITERION: 'spread <= 1.5 ms (max - min) in >= 5 of 6' was written for four replicates; with six rounds max - min grows with n: use the SD or IQR of the replicate medians (e.g. SD <= 0.5 ms) and state it before the run; add the reproduction clause to the VALIDITY rule (E243 had both failures: a session with a good spread but all medians +2 ms is still not quotable), and note that rc1 (e) has no stored value, so its reproduction cannot be checked. (3) VOIDED replicates are skipped and never repeated; the 'all pairs on the same side' clause then has fewer pairs: append one repeat round per void (max 2) or require >= 5 valid pairs for a-c. (4) Prediction as {value, tol, class}: a - c = -0.3 ms (tol 0.7) is an improvement-sign question for decide.py (E243 referee point 3: write +0.3 as improvement or label the verdict manual); the three bundled claims (j1 vs f1, recall +2 - 3 ms, recursion +6 ms) each get their own check in the summary. (5) The TIE cannot fail the idea 'j1 is not faster' by more than the band: state that a -0.5 ms result with all pairs negative (as in E243) is again TIE, and what that means for D4 (comparison at equal quality per ms, not 'fast variant'). (6) 'Other chats asked to stay off the machine' is a request, not a lease: the run must wait for the loop to end (it is planned LAST) and `schedule.py` must hold no other item; no WSL job, no browser, no evaluation during the 45 min. (7) Paths: the card says `analyse_e253.py` in temp/loop20/, the driver header says temp/loop19/analyse_e243.py; fix. Cost 45 min and the 1 h stop are realistic (36 replicates x ~1.2 min; voids add time).

## 3. Summary (2026-10-27)
- **Not run: blocked by the shared machine (decision rule not reached, no verdict, schedule item E253.t stays planned).** The card needs an exclusive machine for 45 minutes. The BEV-system chat started a 7-hour training (`t1_r4_soa`) while loop 20 was running its own trainings and asked that no timing run next to it; the guards of `temp/loop20/run_e253.sh` (GPU use above 5 %, WSL load above 3, Windows CPU above 20 %) would have voided every replicate. A timing taken anyway would repeat the invalid E243 session (N1).
- **What is ready:** the revised card (value / tolerance / class, session validity rule: at least 5 valid replicates per configuration, replicate SD <= 0.6 ms in at least 5 of 6 configurations, stored values reproduced within 1.0 ms), the driver with machine guards and repair rounds, `analyse_e253.py`. When `pgrep -f t1_r4` finds nothing and the load average is below 3: `run.py --id E253.t ... -- bash temp/loop20/run_e253.sh`.
- **Replaced in loop 20 by** E256 (the D4 candidates by city, CPU only). The timing numbers of the week-3 report remain the stored loop-18 values and the ordering of the invalid E243 session.
- **System finding (N1):** an exclusive window agreed by message between two chats was overrun; a lease file with an owner, a start and an end that `run.py` checks (and that the tools chat owns) is the fix. Until then: ask the other chat to hold its start, or run timing only when no other chat has announced a job.
