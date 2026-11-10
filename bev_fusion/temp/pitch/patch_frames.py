def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    ('''    @classmethod
    def all(cls, of: str = "val") -> "Frames":
        return cls("all", (), of)

    @classmethod
    def spread(cls, n: int, of: str = "val") -> "Frames":
        return cls("spread", (int(n),), of)

    @classmethod
    def random(cls, n: int, seed: int, of: str = "val") -> "Frames":
        return cls("random", (int(n), int(seed)), of)

    @classmethod
    def scenes(cls, names: Sequence[str], of: str = "val") -> "Frames":
        return cls("scenes", (tuple(names),), of)

    @classmethod
    def pick(cls, ids: Sequence[str], of: str = "val") -> "Frames":
        return cls("frames", (tuple(ids),), of)

    @classmethod
    def parse(cls, text: "str | Frames") -> "Frames":
        if isinstance(text, Frames):
            return text
        parts = str(text).split(":")
        kind = parts[0]
        if kind not in cls.KINDS or kind == "all":
            return cls.all(str(text))                                           # a bare set name''', '''    @classmethod
    def parse(cls, text: "str | Frames") -> "Frames":
        """a Frames from its text (see the module text); a Frames is returned as it is"""
        if isinstance(text, Frames):
            return text
        parts = str(text).split(":")
        kind = parts[0]
        if kind not in cls.KINDS or kind == "all":
            return cls("all", (), str(text))                                    # a bare set name'''),
    ('''    __repr__ = __str__

    def __eq__(self, other) -> bool:
        return isinstance(other, Frames) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))
''', '''    __repr__ = __str__
'''),
    ('''Frames: a text or a Frames object. `val`''', '''Frames: a text (`Frames.parse`). `val`'''),
    ('''    """which frames of which set: a value; `resolve(source)` -> (the scenes to run, the frames to keep and score)"""''', '''    """which frames of which set, written as the text of the module docstring; `resolve(source)` -> (the scenes to run, the frames to keep and score)"""'''),
])
patch("evalsuite/selftest_api.py", [
    ('''    assert Frames.parse("random:3:small.val") == Frames.random(3, 0, "small.val") and Frames.parse("random:3") == Frames.random(3, 0) and Frames.parse("random:3:7") == Frames.random(3, 7)
''', '''    assert Frames.parse("random:3:small.val").args == (3, 0) and Frames.parse("random:3").args == (3, 0) and Frames.parse("random:3:7").args == (3, 7)
'''),
    ('''    assert Frames.parse("spread:5") == Frames.spread(5) and str(Frames.parse("spread:5")) == "spread:5:val" and str(Frames.parse(Frames.all("x"))) == "x"
    scenes, keep = Frames.all().resolve(src)
    assert scenes == ["demo-0", "demo-1", "demo-2"] and len(keep) == 18
    s5, k5 = Frames.spread(5).resolve(src)''', '''    assert str(Frames.parse("spread:5")) == "spread:5:val" and str(Frames.parse(Frames.parse("x"))) == "x"
    P = Frames.parse
    scenes, keep = P("val").resolve(src)
    assert scenes == ["demo-0", "demo-1", "demo-2"] and len(keep) == 18
    s5, k5 = P("spread:5").resolve(src)'''),
    ('''    a, b = Frames.random(2, 1).resolve(src)[0], Frames.random(2, 1).resolve(src)[0]
    assert a == b and len(a) == 2 and any(Frames.random(2, s).resolve(src)[0] != a for s in range(2, 8)), "random: reproducible with its seed, another seed another scenes"
    assert Frames.random(2, 1).resolve(src)[1] == [r for s in a for r in src.frames(s)], "random scenes keep all their frames"
    assert [r.label for r in Frames.pick(["demo-2/5", "demo-0/1"]).resolve(src)[1]] == ["demo-0/001", "demo-2/005"]''', '''    a, b = P("random:2:1").resolve(src)[0], P("random:2:1").resolve(src)[0]
    assert a == b and len(a) == 2 and any(P(f"random:2:{s}").resolve(src)[0] != a for s in range(2, 8)), "random: reproducible with its seed, another seed another scenes"
    assert P("random:2:1").resolve(src)[1] == [r for s in a for r in src.frames(s)], "random scenes keep all their frames"
    assert [r.label for r in P("frames:demo-2/5,demo-0/1").resolve(src)[1]] == ["demo-0/001", "demo-2/005"]'''),
    ('''    _raises(EvaluationError, "not in val", Frames.scenes(["demo-9"]).resolve, src)
    _raises(EvaluationError, "not a frame", Frames.pick(["demo-0/99"]).resolve, src)
    _raises(EvaluationError, "not a frame", Frames.pick(["junk"]).resolve, src)''', '''    _raises(EvaluationError, "not in val", P("scenes:demo-9").resolve, src)
    _raises(EvaluationError, "not a frame", P("frames:demo-0/99").resolve, src)
    _raises(EvaluationError, "not a frame", P("frames:junk").resolve, src)'''),
    ('''_engine(net3, Frames.all()), "resumed run")''', '''_engine(net3, Frames.parse("val")), "resumed run")'''),
])
print("ok")
