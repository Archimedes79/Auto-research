p = "temp/pitch/split_counts.py"
t = open(p, encoding="utf-8").read()
a = '''"num_lidar_pts": (\d+), "num_radar_pts": (\d+)'''
assert t.count(a) == 1, t.count(a)
t = t.replace(a, '''"num_lidar_pts": (\d+),\s*"num_radar_pts": (\d+)''')
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("regex fixed")
