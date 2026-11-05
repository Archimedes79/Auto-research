for t in 30f65d58c9c34488991774bec0ff0500 5bb38a1b262649debccfd4fc00c1c600 c28b310e5a3f4f258f1ef52e0f3c3700; do
  grep -m1 -A40 -P "\"token\": \"$t\"" ~/data/nuscenes_full/v1.0-trainval/sample_annotation.json | grep -E 'num_lidar_pts|num_radar_pts' | tr -d '\n ' ; echo " <- $t"
done
