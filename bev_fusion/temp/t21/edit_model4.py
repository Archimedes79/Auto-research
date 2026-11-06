p = "training/model.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''    def _unroll(self, frames, steps, cam_state, step, obs, buf=None):''', '''    @staticmethod
    def _step_to_oldest(gap, steps, frames):
        """the trainer's keyframe gap step (dt, v, omega) of the gap mean as the step to the OLDEST frame: dt less the time of the oldest
        frame's own chain (clamped at 0), v and omega kept"""
        dt, v, w = gap
        used = sum((s[0] for s in as_chain(steps[-1], frames)), frames.new_zeros(frames.shape[0]))
        return (dt - used).clamp(min=0), v, w

    def _unroll(self, frames, steps, cam_state, step, obs, buf=None):''')
rep('''        oldest frame: stream (one frame) as given (a triple or the sweep chain since the previous frame); batch the trainer's keyframe gap
        step (dt, v, omega of the gap mean, one triple), whose dt loses the time of the oldest frame's own chain (clamped at 0), v and
        omega kept. Needs''', '''        oldest frame: stream (one frame) as given (a triple or the sweep chain since the previous frame); batch the trainer's keyframe gap
        step (one triple) reduced by `_step_to_oldest`. Needs''')
rep('''            dt, v, w = move[0]
            used = sum((s[0] for s in as_chain(steps[-1], frames)), frames.new_zeros(B))
            move = [((dt - used).clamp(min=0), v, w)]
''', '''            move = [self._step_to_oldest(move[0], steps, frames)]
''')
open(p, "w", encoding="utf-8", newline="").write(s)
