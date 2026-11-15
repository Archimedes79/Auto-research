"""scratch: the gradient check of section J compares the sets of tensors that receive a gradient (not the count of all trainable tensors of the model)"""
p = "tests/test_student_chain.py"
s = open(p, encoding="utf-8").read()
old = ("check(set(g_live) == set(g_ref) and len(g_live) >= n_trainable - 2 and all(torch.equal(g_live[n], g_ref[n]) for n in g_ref),\n"
       "      f\"... and so does the gradient of each of the {len(g_ref)} tensors that receive one (student, lift tables, camera reduction)\")\n")
new = ("student_named = [\"student.\" + n for n, p_ in st.named_parameters() if p_.requires_grad]\n"
       "check(set(g_live) == set(g_ref) and all(n in g_ref for n in student_named) and all(torch.equal(g_live[n], g_ref[n]) for n in g_ref),\n"
       "      f\"... and so does the gradient of each of the {len(g_ref)} tensors that receive one (all {len(student_named)} of the student, the camera reduction)\")\n")
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("n_trainable = sum(1 for p in tr.parameters() if p.requires_grad)\n", "")
open(p, "w", encoding="utf-8").write(s)
print("ok")
