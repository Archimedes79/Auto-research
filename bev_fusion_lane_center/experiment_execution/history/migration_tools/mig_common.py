"""Shared definitions of the folder migration (plan + apply use the same rules).
Old root O = /mnt/c/ML-Experiment, new project root R = O/bev_fusion_lane_center."""
import json
import os
import re

O = "/mnt/c/ML-Experiment"
R = f"{O}/bev_fusion_lane_center"
STAGE = f"{R}/_to_delete"  # nothing is purged: staged items wait for the user's OK
RS = f"{O}/restructure"

JUNK_RE = re.compile(r"^(_patch|_rv_|_fix|_fx|_dbg|dbg|_notes|_q\d|_p\d|_chk|_fill|_add|_jl|_show|_dump|_inspect|_mem|_update|_gen|_mkcards|_decide|_survey|_recover|_scan|_replay|_cmp|_try|_tmp|test_|_kill|_wait|_restart|_start|_relaunch|_renice|_pause|_stop|_bg|_ps\b|_queue|queue_|launch_|kill_|_enq|_resume|_unlock|_noshutdown|_peek|peek_|_ls_|debug_|diag_nan|post_watch|state_check|_count_checks|_syntax|_sig)")
JOBCTL_RE = re.compile(r"^(_kill|_wait|_restart|_start|_relaunch|_renice|_pause|_stop|_bg|_ps\.|_queue|queue_|launch_|kill_|_enq|_resume|post_watch)")
BAK_RE = re.compile(r"\.(bak|bak\d*|bak_[A-Za-z0-9_]+|lock|state|closed|out)$")

# archive groups for historic scripts that are neither live nor assigned to an experiment id
ARCHIVE_RULES = [
    (r"^(setup_|crn_env|fix_dns|fix_ipv4|check_env|gcar_nvcc|gcar_build_rast|gcar_patch_rast|patch_bevfusion|download_nuscenes|extract_blob|crn_fetch|_install|install_|netspeed|inventar|lidar_inventory|delete_lidar|run_py|_py\.sh|_pyr)", "tools_env"),
    (r"^(crn_|run_crn|make_report_crn|make_reports_crn|viz_crn|make_crn|run_unet|run_scaling|run_stage|run_bevfusion|run_lidar|run_gridbase|run_k\.sh|run_extract|run_pca|run_floor|run_post|run_main_dense|lidar_|extract_features|pca_features|fix_infos|run_v\d|run_wide|run_wrong|run_strict|run_odo|run_align|run_btrain)", "loop01_foundation_runs"),
    (r"^(build_final|build_headline|build_loop\d|build_all|build_e2e|build_decision|build_proposal|build_experiment_table|final_text|make_report|make_loop\d+_tables|make_overview|make_arch|make_final_arch|make_intro|regen_old|finish_report|write_readmes|report2|fig_p[89]|fig_p1[0-3]|fig_decision|fig_e|make_cards|summ\.py|nums\.py|final_summary|headline|text_e|loop\d+_text|master_final|_final_|_finalize|_assemble|_montage|_mk_readmes|_readme|_html2txt|_pdf|_find|_dump_c|md_to_html)", "reports_build"),
    (r"^(_build_bundle|_stage_bundle|_run_bundle|bundle\d|_bundle_|_make_build|_split_zip|_serve_mail|_rezip|_run_build|_run_p13|_run_p14|_run_p15)", "bundles_and_run_commands"),
    (r"^(_decide|_final16|_state_|_phase16|_next1|_plan_r0|_table_r|_r3tab|_run_r3|_queue_r)", "state_writers"),
    (r"^(check_|sanity_|diag_|dense_ext|est_|eval_|combo_|sweep_|choose_|pick_|range_|oracle_|policy_|path_eval|jump_eval|leak_|fold_|boot_loc|calib_|stale_|scene_bias|slot_test|monitor_)", "diagnostics_and_sweeps"),
]
ARCHIVE_RES = [(re.compile(p), g) for p, g in ARCHIVE_RULES]


def archive_group(name):
    for rx, g in ARCHIVE_RES:
        if rx.match(name):
            return g
    return "misc"


def slug(s, n=6):
    s = re.sub(r"\(.*?\)", " ", s)
    w = re.findall(r"[A-Za-z0-9]+", s.lower())
    stop = {"the", "a", "an", "of", "and", "on", "for", "with", "to", "in", "by", "as", "is", "are", "at"}
    w = [x for x in w if x not in stop][:n]
    return "-".join(w) or "experiment"


def ids_of_row(idstr):
    """'E49 / E50 / E52' -> [49,50,52]; 'E67 - E72' -> 67..72; 'E120a - E120c' -> [120]; 'VLA' -> []"""
    nums = []
    parts = re.split(r"\s*/\s*", idstr)
    for p in parts:
        m = re.match(r"E(\d+)[a-z]?\s*-\s*E?(\d+)[a-z]?$", p.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            nums += list(range(a, b + 1)) if 0 <= b - a <= 12 else [a, b]
        else:
            nums += [int(x) for x in re.findall(r"E(\d+)", p)]
    return sorted(set(nums))


def folder_name(row):
    """E<first id, three digits><letter if any>_<slug of the description>; the other ids of a row are listed in its README"""
    m = re.search(r"E(\d+)([a-z]?)", row["id"])
    tag = f"E{int(m.group(1)):03d}{m.group(2)}" if m else (re.sub(r"[^A-Za-z0-9]+", "", row["id"]) or "X")
    return f"{tag}_{slug(row['what'])}"


def load_table():
    t = json.load(open(f"{O}/out/loop2/EXPERIMENT_TABLE.json", encoding="utf-8"))
    return t["rows"]


def eid_of_name(name):
    m = re.match(r"^(?:e|sys_e|tok_e|route_test_e|queue_e|rowloss_e|sweep_trace_e)(\d+)", name)
    return int(m.group(1)) if m else None


def eids_in_script(name):
    return [int(x) for x in re.findall(r"(?:^|[_\-])e(\d{2,3})(?=[_\.a-z]|$)", name)]
