f="/mnt/c/Program Files/DownloadHelper CoApp/ffmpeg.exe"
[ -x "$f" ] || { echo "no ffmpeg at $f"; exit 1; }
[ -e ~/.evalsuite_ffmpeg ] || echo "$f" > ~/.evalsuite_ffmpeg
cat ~/.evalsuite_ffmpeg
cd "$1" && CUDA_VISIBLE_DEVICES= /home/ml/miniconda3/envs/gcar/bin/python -c "from evalsuite.render.video import find_ffmpeg; print('found:', find_ffmpeg())"
