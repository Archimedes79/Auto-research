p = "autoresearch/evalsuite/comparison/maps.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def _disc(radius_cells: float) -> np.ndarray:
    """the structuring element of a dilation: (2r + 1)^2 booleans, set where the offset (dx, dy) lies within radius_cells"""
    r = int(math.floor(radius_cells + 1e-9))
''', '''def _cells(radius_cells: float) -> int:
    """the whole cells of a radius (an integer up to float noise, like 0.3 / 0.1, is that integer)"""
    return int(math.floor(radius_cells + 1e-9))


def _disc(radius_cells: float) -> np.ndarray:
    """the structuring element of a dilation: (2r + 1)^2 booleans, set where the offset (dx, dy) lies within radius_cells"""
    r = _cells(radius_cells)
''', 1)
s = s.replace('''    r = int(math.floor(radius_cells + 1e-9))
    rows, cols = m.shape''', '''    r = _cells(radius_cells)
    rows, cols = m.shape''', 1)
s = s.replace("    if int(math.floor(radius_cells + 1e-9)) <= 0:", "    if _cells(radius_cells) <= 0:", 1)
open(p, "w", encoding="utf-8").write(s)
p = "autoresearch/evalsuite/comparison/mutation_check.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''("maps.py", "r = int(math.floor(radius_cells + 1e-9))", "r = int(math.floor(radius_cells))",''', '''("maps.py", "return int(math.floor(radius_cells + 1e-9))", "return int(math.floor(radius_cells))",''', 1)
open(p, "w", encoding="utf-8").write(s)
