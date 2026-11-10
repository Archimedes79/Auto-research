"""One-off: list the trainval archives of the public nuScenes bucket with size and ETag (listing only, no download).

Run from the problem folder: wsl python3 temp/loop19/list_nuscenes_bucket.py [filter]
"""
import re
import sys
import urllib.parse
import urllib.request

BUCKET = "https://motional-nuscenes.s3.amazonaws.com/"
flt = sys.argv[1] if len(sys.argv) > 1 else "trainval"
token, rows = None, []
while True:
    q = {"list-type": "2", "max-keys": "1000"}
    if token:
        q["continuation-token"] = token
    xml = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60).read().decode()
    for key, etag, size in re.findall(r"<Key>([^<]*)</Key>.*?<ETag>([^<]*)</ETag><Size>(\d+)</Size>", xml):
        if flt in key:
            rows.append((key, int(size), etag.replace("&quot;", "")))
    m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
    if not m:
        break
    token = m.group(1)
for key, size, etag in rows:
    print(f"{size / 1e9:8.2f} GB  {key}  {etag}")
print(f"{sum(s for _, s, _ in rows) / 1e9:8.2f} GB  total ({len(rows)} files)")
