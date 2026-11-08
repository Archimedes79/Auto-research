p = "evaluation/bev_response.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    allb = [[dict(b, score=1.0) for b, in [(b,)] if "cls" in b] for bs, _ in ann]

    def strip(bs):
        return [{k: v for k, v in b.items() if k != "cls"} for b in bs]
''', '''
    def strip(bs):
        return [dict({k: v for k, v in b.items() if k != "cls"}, score=1.0) for b in bs]
''')
s = s.replace('''    m = det_numbers(samples, [[dict(b, score=1.0) for b in strip(bs)] for bs, _ in ann], task, "det")
    lab = "det (old) on the same detections: the unobserved boxes' detections are false alarms, their GT is not counted"
    rows += [(lab, "n_det - n_gt of det = detections on unobserved boxes >= 0", float(m[f"{CLASS}/all/n_det"] - m[f"{CLASS}/all/n_gt"] >= 0), 1.0, 0.0)]
''', '''    m = det_numbers(samples, everything, task, "det")
    n_all = sum(1 for bs, _ in ann for b in bs if b["group"] == CLASS)
    lab = "det (old) on the same detections: nothing is ignored, the detections on unobserved boxes are false alarms (the known D38 gap)"
    rows += [(lab, "n_det = every detection in the window", m[f"{CLASS}/all/n_det"], n_all, 0.0), (lab, "n_gt < every annotated box (lidar-based `observed`)", float(m[f"{CLASS}/all/n_gt"] <= n_all), 1.0, 0.0)]
''')
s = s.replace("    from evaluation.bev_fusion_area import COV_CLASSES\n", "")
s = s.replace('''    return samples, (boxes_rows(samples, task) + maps_rows(samples, task) + velocity_rows(samples, task)
                     + swap_boxes_rows(samples, task) + swap_maps_rows(samples, task) + swap_velocity_rows(samples, task))''', '''    return samples, (boxes_rows(samples, task) + maps_rows(samples, task) + velocity_rows(samples, task) + fusion_rows(samples, task)
                     + swap_boxes_rows(samples, task) + swap_maps_rows(samples, task) + swap_velocity_rows(samples, task))''')
open(p, "w", encoding="utf-8").write(s)
p = "evaluation/bev_gtcheck.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    task.curves = True\n    return task", "    task.curves = True\n    task.fusion = True\n    return task")
open(p, "w", encoding="utf-8").write(s)
