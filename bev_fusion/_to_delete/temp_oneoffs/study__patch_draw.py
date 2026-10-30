p = "temp/study/draw_architecture_t3.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''ax.add_patch(FancyArrowPatch((147, 74), (74, 55.5), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="0.25", connectionstyle="arc3,rad=0.0"))''',
'''ax.plot([147, 147, 72], [74, 60, 60], color="0.25", lw=1.1)
ax.add_patch(FancyArrowPatch((72, 60), (72, 55.6), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="0.25"))''')
s = s.replace('''alt(128, y0, 18.5, "none: TIE (0.321)", "tie", h=4.6)
alt(147.5, y0, 18.5, "ego warp + ConvGRU: TIE (0.327)", "tie", h=4.6)''', '''alt(128, y0, 18, "no temporal block:\nTIE (0.321)", "tie")
alt(148, y0, 18, "ego warp + ConvGRU:\nTIE (0.327)", "tie")''')
s = s.replace('''alt(90, y1, 34, "random init: WORSE (walkway -0.055*)\nSECOND + SECONDFPN, width 0.5, forward splat: not run yet", "todo")''',
'''alt(90, y1, 16.5, "random init: WORSE\n(walkway -0.055*)", "worse")
alt(107.5, y1, 16.5, "SECOND, width 0.5,\nforward splat: not run", "todo")''')
s = s.replace('''alt(130, y1 if False else 34.5, 36, "velocity head''', '''alt(130, 34.5, 36, "velocity head''')
s = s.replace('xx = 82 + i * 22', 'xx = 70 + i * 24')
s = s.replace('"outline of an alternative: ", fontsize=8.5', '"outline of an alternative:", fontsize=8.5')
s = s.replace('(("better", "BETTER"), ("tie", "TIE (inside the noise band)"), ("worse", "WORSE"), ("todo", "not run / running"))', '(("better", "BETTER"), ("tie", "TIE (in the band)"), ("worse", "WORSE"), ("todo", "not run / running"))')
s = s.replace('ax.text(2, 10, "fill: grey = frozen / cached, blue = trained, green = outputs      outline of an alternative:"', 'ax.text(2, 10, "fill: grey = frozen, blue = trained, green = outputs;  outline:"')
open(p, "w", encoding="utf-8", newline="\n").write(s)
