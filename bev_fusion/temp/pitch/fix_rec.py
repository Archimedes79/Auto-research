p = "evaluation/bev_modules.py"
s = open(p, encoding="utf-8").read()
old = '''            if m is not None and name != "trunk":
                self._hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: self.store.setdefault(n, tensors_of(o)) and None))'''
new = '''            if m is not None and name != "trunk":
                self._hooks.append(m.register_forward_hook(self._hook(name)))'''
assert old in s
s = s.replace(old, new)
old = '''    def clear(self):
        self.store.clear()
'''
new = '''    def _hook(self, name):
        def hook(mod, inputs, output):                                  # returns None: a hook that returns a value REPLACES the module's output
            self.store.setdefault(name, tensors_of(output))
        return hook

    def clear(self):
        self.store.clear()
'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
