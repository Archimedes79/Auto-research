"""scratch: run the round-4 mutants one by one against the tests that guard each rule (a full-suite run per mutant takes about 45 minutes on the loaded box), half of them per process: run_mut5.py 0|1"""
import os, subprocess, sys
R = os.environ["BFLC_ROOT"]
PIN = "test_the_code_pin_cannot_be_bypassed,test_expect_code_digest_refuses_other_code_before_any_work,test_the_test_line_is_fixed_by_the_card"
MAP = [("pin ", PIN), ("environment", PIN), ("undigested", "test_undigested_modules_the_environment_and_the_repository_modules"), ("repo_modules", "test_undigested_modules_the_environment_and_the_repository_modules"),
       ("gpu gate", "test_a_dump_on_a_gpu_waits_for_room_like_jends_eval"), ("TEST artefact", "test_on_test_a_flagged_dump_without_provenance_is_one_of_e321s_artefacts_and_rows_are_validated_in_the_precheck"),
       ("precheck rows", "test_on_test_a_flagged_dump_without_provenance_is_one_of_e321s_artefacts_and_rows_are_validated_in_the_precheck"),
       ("precheck default", "test_a_flagged_dump_with_an_old_style_head_file_is_exempt_and_refusals_come_in_the_precheck"), ("precheck seed unchecked", "test_a_flagged_dump_with_an_old_style_head_file_is_exempt_and_refusals_come_in_the_precheck"),
       ("message interrupted", "test_a_flagged_dump_with_an_old_style_head_file_is_exempt_and_refusals_come_in_the_precheck"), ("early", "test_the_control_is_checked_before_any_new_dump"),
       ("TEST boot", "test_the_test_line_is_fixed_by_the_card"), ("TEST seeds", "test_the_test_line_is_fixed_by_the_card"), ("TEST b_dump", "test_the_test_line_is_fixed_by_the_card"), ("TEST control", "test_the_test_line_is_fixed_by_the_card"),
       ("TEST scenes", "test_the_test_line_is_fixed_by_the_card"), ("sidecar not first", "test_dump_b_and_dump_ends_on_a_stub_model"), ("pickle overwrite", "test_dump_b_and_dump_ends_on_a_stub_model"),
       ("check_seed", "test_precheck_refuses_before_any_dump_is_made"), ("score_seed", "test_score_seed_control_checks"), ("tol REHEARSAL", "test_score_seed_control_checks"), ("expect", "test_expect_code_digest_refuses_other_code_before_any_work"),
       ("cmd_all no gate", "test_the_test_gate_guards_every_entry"), ("main dump no gate", "test_the_test_gate_guards_every_entry"), ("e321_changed", "test_e321_guard_detects_a_change_and_a_missing_file"), ("guard_e321", "test_e321_guard_detects_a_change_and_a_missing_file"),
       ("atomic", "test_write_helper_refuses_e321_and_is_atomic"), ("is_protected", "test_write_helper_refuses_e321_and_is_atomic"), ("get_dump flagged", "test_a_dump_is_reused_only_under_the_same_provenance"),
       ("provenance line removed", "test_provenance_is_taken_before_the_model_is_loaded")]
names = ['pin run prefix not checked', 'pin rehearsal ok not checked', 'pin drift not checked', 'pin digest of file list not checked', 'pin card digest ignored', 'pin environment record not checked', 'environment compare off', 'environment_diff versions ignored', 'undigested not computed', 'undigested includes preloaded', 'repo_modules temp kept', 'repo_modules GtGenerator kept', 'repo_modules relative kept', 'gpu gate removed (dump_b)', 'gpu gate removed (dump_ends)', 'gpu gate always (dump_b)', 'gpu gate always (dump_ends)', 'TEST artefact rule off', 'precheck rows not validated', 'early control off', 'early counts scored again', 'TEST boot not fixed', 'TEST seeds not fixed', 'TEST b_dump not required', 'TEST control_ends not required', 'TEST control_card not required', 'TEST scenes not checked', 'sidecar not first (dump_b)', 'sidecar not first (dump_ends)', 'pickle overwrite allowed (dump_b)', 'pickle overwrite allowed (dump_ends)', 'check_seed seed optional']
part = int(sys.argv[1])
for i, n in enumerate(names):
    if i % 2 != part:
        continue
    tests = next((t for p, t in MAP if n.startswith(p)), "")
    assert tests, n
    subprocess.run(["bash", os.path.join(R, "tools", "env", "_py14.sh"), os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py"), n],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MUT_TESTS=tests), stdout=sys.stdout, stderr=sys.stderr)
