p = "autogt/nuscenes_gt.py"
s = open(p, encoding="utf-8").read()
s = s.replace("        vel = np.zeros((2,) + R.shape, np.float32)\n        sig = np.zeros(R.shape, np.float32)\n",
              "        vel = np.zeros((2,) + R.shape, np.float32)\n        sig = np.zeros(R.shape, np.float32)\n        pres = R.canvas()                                        # sub-pixels covered by a box with known velocity\n")
s = s.replace("                if known:\n                    vel[0][one > 0], vel[1][one > 0] = ve[0], ve[1]\n                    sig[one > 0] = vsig\n",
              "                if known:\n                    vel[0][one > 0], vel[1][one > 0] = ve[0], ve[1]\n                    sig[one > 0] = vsig\n                    pres |= one\n")
s = s.replace('        out["velocity"] = np.stack([R.at_centres(vel[0]), R.at_centres(vel[1])])\n',
              '        # velocity at a cell = the box velocity averaged over the covered central sub-pixels only (an edge cell keeps the\n'
              '        # full speed of its box instead of a coverage-scaled one)\n'
              '        pc = R.at_centres(pres.astype(np.float32))\n'
              '        out["velocity"] = (np.stack([R.at_centres(vel[0]), R.at_centres(vel[1])]) / np.maximum(pc, 1e-6)\n'
              '                           * (pc > 0)).astype(np.float32)\n')
open(p, "w", encoding="utf-8").write(s)
p = "autogt/selftest.py"
s = open(p, encoding="utf-8").read()
s = s.replace('    cells = k["instance"] == moving[0]\n    assert np.allclose(k["velocity"][:, cells], bx[moving[0]]["vel"][:, None], atol=1e-4)',
              '    cells = (k["instance"] == moving[0]) & (k[bx[moving[0]]["group"]] > 0)\n    assert cells.any() and np.allclose(k["velocity"][:, cells], bx[moving[0]]["vel"][:, None], atol=1e-3)   # edge cells too')
open(p, "w", encoding="utf-8").write(s)
print("ok")
