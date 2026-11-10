"""scratch: section J also checks that every lift call (recomputations included) gets a projection-table key made in the forward"""
p = "tests/test_student_chain.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub("tr.lift.forward = lambda *a, **k: (lift_calls.append(1), lift_fwd(*a, **k))[1]\n\n\ndef counted(",
    "tr.lift.forward = lambda *a, **k: (lift_calls.append(k.get(\"key\")), lift_fwd(*a, **k))[1]\n\n\ndef counted(")
sub("    st.checkpoint_encode = True\n    return len(calls), len(lift_calls)\n",
    "    st.checkpoint_encode = True\n    return len(calls), len(lift_calls), list(lift_calls)\n")
sub("n_enc, n_lift = counted(True)\ncheck((n_enc, n_lift) == (2, 6),",
    "n_enc, n_lift, keys_seen = counted(True)\ncheck((n_enc, n_lift) == (2, 6),")
sub("n_enc, n_lift = counted(True, student_flag=False)\n", "n_enc, n_lift, _ = counted(True, student_flag=False)\n")
sub("n_enc, n_lift = counted(False)\n", "n_enc, n_lift, _ = counted(False)\n")
sub("check((n_enc, n_lift) == (1, 3), f\"evaluation mode",
    "check(all(k_ is not None for k_ in keys_seen), \"every lift call, the recomputations included, gets the projection-table key made once in the forward (no wait for the GPU inside a backward)\")\n"
    "check((n_enc, n_lift) == (1, 3), f\"evaluation mode")
open(p, "w", encoding="utf-8").write(s)
print("ok")
