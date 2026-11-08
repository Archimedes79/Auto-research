"""scratch: text of a html report (tags stripped, block tags -> newlines) for proofreading. usage: html2txt.py in.html out.txt"""
import html
import re
import sys

s = open(sys.argv[1], encoding="utf-8").read()
s = re.sub(r"<style.*?</style>", "", s, flags=re.S)
s = re.sub(r"<svg.*?</svg>", "[chart]", s, flags=re.S)
s = re.sub(r"<img[^>]*>", "[image]", s)
s = re.sub(r"<video[^>]*>.*?</video>", "[video]", s, flags=re.S)
s = re.sub(r"</(p|div|h\d|li|tr|table|ul|details|summary|figure)>", "\n", s)
s = re.sub(r"</t[dh]>", " | ", s)
s = re.sub(r"<[^>]+>", "", s)
s = html.unescape(s)
s = re.sub(r"\n\s*\n+", "\n", s)
open(sys.argv[2], "w", encoding="utf-8").write(s)
print(len(s), "characters,", s.count("\n"), "lines")
