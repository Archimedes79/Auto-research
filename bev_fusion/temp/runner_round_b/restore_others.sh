#!/bin/bash
# scratch: move back everything from temp/runner_round that I did not create (I created exactly the names in MINE or matching the patterns)
cd /mnt/c/Auto-research/bev_fusion/temp || exit 1
MINE=" DECISIONS_fixed.md add_rows.py blobs blobs_runner.py blobs_runsuite.py check_diffs.sh check_hook.sh check_jobs.sh check_tools.sh close_four.sh close_old.sh commit_close.sh commit_close_msg.txt
 commit_d67.sh commit_d67_msg.txt commit_d67b.sh commit_d67b_msg.txt commit_d70.sh commit_d70_msg.txt commit_d72.sh commit_d72_msg.txt commit_d73.sh commit_d73b.sh commit_evalreadme.sh
 commit_evalreadme_msg.txt commit_free.sh commit_free2.sh commit_free2_msg.txt commit_free3.sh commit_free3_msg.txt commit_free_msg.txt commit_hook.sh commit_hook_msg.txt commit_init.sh
 commit_init_msg.txt commit_jobs.sh commit_jobs_msg.txt commit_runar.sh commit_runar2.sh commit_runner.sh commit_runner_msg.txt commit_runsuite.sh commit_runsuite_msg.txt d73_msg.txt d73b_msg.txt
 diff_runner.sh edit_conform.py edit_jobs_readmes.py edit_selftest.py fix_decisions.py park_draft.sh probe_cli.sh rename_results.sh rename_results2.sh rename_runjobs.sh repair_commit.sh
 repair_d68.sh repair_msg.txt revert_readme_mine.py run_conform.sh run_frozen.sh run_real_runner.sh run_rs_selftest.sh run_test_free.sh run_test_free2.sh run_test_free3.sh run_test_free4.sh
 run_test_pred.sh run_test_runner.sh run_tests_runner.sh run_ts_selftest.sh runar_msg.txt runner_draft runsuite_api_consolidated.py runsuite_api_head.py runsuite_layers_2026-11-10 set_done.sh
 show_holds.py speed_free.py suite_methods.sh sync_rows.py test_free.log test_free2.log test_free3.log test_free4.log test_pred.log test_real.log try_cli.sh try_import.sh try_m_runner.sh
 try_runner.sh ts_selftest.log ts_selftest_api.log ts_selftest_api2.log wait_log.sh scan_area.sh unused_scan.py run_unused_scan.sh run_keep3_real.sh verify_readme_cmds.sh id_compare.sh
 show_methods.sh check_hook.sh edit_tests_round.py run_base_real.sh run_t16_again.sh edit_real_test.py wait_proc.sh move_mine.sh t20_val8.json base_t16_again.log "
back=0
cd runner_round || exit 1
for f in * .[!.]*; do
  [ -e "$f" ] || continue
  case "$MINE" in *" $f "*) continue;; esac
  case "$f" in keep3_*.log|base_*.log) continue;; esac
  mv "$f" ../cleanup/ && { back=$((back + 1)); echo "back: $f"; }
done
echo "moved back $back entries"
