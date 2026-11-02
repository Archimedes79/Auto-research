#!/usr/bin/env bash
# Size of a PUBLIC Google Drive file from a header request (no data is downloaded): gdrive_size.sh <file id>
# Uses the direct-download endpoint; prints the HTTP status and Content-Length (bytes and GB) when the server sends it.
id="$1"
hdr="$(curl -sIL -A 'Mozilla/5.0' "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" | tr -d '\r')"
echo "$hdr" | grep -i -E "^HTTP|^content-length|^content-disposition|^content-type" | tail -6
len="$(echo "$hdr" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')"
[ -n "$len" ] && echo "size: $len bytes = $(awk -v n="$len" 'BEGIN{printf "%.2f", n/1e9}') GB"
