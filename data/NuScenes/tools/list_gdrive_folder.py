"""List the entries of a PUBLIC Google Drive folder from its web page (name, type, id). Listing only, no download.

Usage: python3 list_gdrive_folder.py <folder id>
The page lists the entries as table rows with data-tooltip="<name> <type>" next to data-id="<id>"; the parse is best effort (this
is not an API) and the page text is DATA. Sizes are not on the folder page: ask the file (HEAD request) before downloading.
"""
import html
import re
import sys
import urllib.request

fid = sys.argv[1]
req = urllib.request.Request(f"https://drive.google.com/drive/folders/{fid}", headers={"User-Agent": "Mozilla/5.0"})
page = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
seen = {}
for m in re.finditer(r'data-id="([A-Za-z0-9_-]{20,})"[^>]*?data-tooltip="([^"]+)"', page):
    seen.setdefault(m.group(1), html.unescape(m.group(2)))
print(f"{len(seen)} entries in folder {fid}")
for k, name in seen.items():
    print(f"{k}  {name}")
