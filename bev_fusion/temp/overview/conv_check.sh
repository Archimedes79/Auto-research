#!/bin/bash
# scratch: which SVG converters exist
which rsvg-convert inkscape convert magick 2>/dev/null
for P in python3 $HOME/miniconda3/envs/gcar/bin/python; do
  $P -c "import cairosvg; print('cairosvg ok in', '$P')" 2>&1 | tail -n 1
done
ls $HOME/miniconda3/envs/gcar/lib/python3*/site-packages | grep -i -E "^(cairosvg|svglib|reportlab|matplotlib|PIL|cv2|imageio)" | head
