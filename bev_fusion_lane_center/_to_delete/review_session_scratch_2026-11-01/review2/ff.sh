echo "FFMPEG=$FFMPEG"; which ffmpeg ffmpeg.exe
cd /mnt/c/Auto-research/autoresearch
CUDA_VISIBLE_DEVICES= /home/ml/miniconda3/envs/gcar/bin/python -c "import sys; sys.path.insert(0,'.'); from evalsuite.render import video; print('found:', video.find_ffmpeg())"
grep -rn "FFMPEG" ~/.bashrc 2>/dev/null | head -3
