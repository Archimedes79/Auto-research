import re
p = "training/model.py"
s = open(p, encoding="utf-8").read()
n0 = len(s)
# imports (both styles)
s = s.replace("    from .radar_bev import RadarBEV\n", "    from .radar_bev import RadarBEV\n    from .radar_encoder import RadarEncoder, n_channels\n", 1)
s = s.replace("    from radar_bev import RadarBEV\n", "    from radar_bev import RadarBEV\n    from radar_encoder import RadarEncoder, n_channels\n", 1)
# the slot table
s = s.replace('''          "pillars": lambda grid, c: RadarPillars(grid, c)}''', '''          "pillars": lambda grid, c: RadarPillars(grid, c),
          "delta_slots": lambda grid, c: RadarEncoder(grid, C=c)}      # radar_encoder.py: delta-coded slot raster, pass-through start, side loss''', 1)
# config comment
s = s.replace('''radar_kind: str = "raster"         # "raster" (means + 2 convs) | "pillars" (point MLP + scatter-max, blocks_new.py) | "off" (zeros)''',
              '''radar_kind: str = "raster"         # "raster" (means + 2 convs) | "pillars" (point MLP + scatter-max, blocks_new.py) | "off" (zeros)
                                       # | "delta_slots" (radar_encoder.RadarEncoder: lossless delta-coded slot raster + 2 convs; needs c_rad >= 61;
                                       # its reconstruction side loss is switched on by TrainConfig.radar_aux)''', 1)
open(p, "w", encoding="utf-8").write(s)
print("model.py changed by", len(s) - n0)
