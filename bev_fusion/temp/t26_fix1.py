import re
p = "tests/test_student_window.py"
s = open(p).read()
s = s.replace("""            m.train()
            tr, tr2 = m.camera_bevs(jb), m.camera_bevs(jb)""", """            train_bn_eval(m)                                                                # training mode (the jitter is on), BatchNorm of the lift fixed: the frames do not couple through their statistics
            tr, tr2 = m.camera_bevs(jb), m.camera_bevs(jb)""")
s = s.replace("""        ok[f"model ({frames}): in training mode the jitter changes frame 0 {'and the older frames' if frames == 'all' else 'only (the cached older frames are the teachers\'s)'}"] = \\""",
              """        ok[f"model ({frames}): in training mode the jitter changes frame 0 and, with 'all', the older frames - with 'current' they stay the cached teachers'"] = \\""")
s = s.replace("""def jitter_model_checks():
    ok = {}""", """def train_bn_eval(m):
    \"\"\"the model in training mode with every BatchNorm in eval mode (the student's already are): a frame's BEV is then independent of the other frames\"\"\"
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def jitter_model_checks():
    ok = {}""")
s = s.replace('not torch.allclose(res_j["p16"], res_c["p16"])', 'not torch.allclose(res_j["c16"], res_c["c16"])')
open(p, "w").write(s)
