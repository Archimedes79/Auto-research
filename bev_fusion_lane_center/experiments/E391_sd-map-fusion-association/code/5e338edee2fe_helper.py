def sub(path, old, new, count=1):
    s = open(path, encoding='utf-8', newline='').read()
    assert s.count(old) >= 1, (path, old[:80])
    s = s.replace(old, new, count)
    open(path, 'w', encoding='utf-8', newline='').write(s)
