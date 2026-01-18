# utils/pillar.py
import numpy as np
from config import *

def create_pillars(points):
    x_min, x_max = POINT_RANGE["x"]
    y_min, y_max = POINT_RANGE["y"]

    x_size = PILLAR_SIZE["x"]
    y_size = PILLAR_SIZE["y"]

    grid_x = int((x_max - x_min) / x_size)
    grid_y = int((y_max - y_min) / y_size)

    pillars = {}

    for p in points:
        x, y = p[0], p[1]
        ix = int((x - x_min) / x_size)
        iy = int((y - y_min) / y_size)

        if 0 <= ix < grid_x and 0 <= iy < grid_y:
            pillars.setdefault((ix, iy), []).append(p)

    pillar_features = []
    coords = []

    for (ix, iy), pts in list(pillars.items())[:MAX_PILLARS]:
        pts = np.array(pts)

        # 🔥 padding 핵심
        if pts.shape[0] < MAX_POINTS_PER_PILLAR:
            pad_len = MAX_POINTS_PER_PILLAR - pts.shape[0]
            pad = np.zeros((pad_len, 4))
            pts = np.vstack([pts, pad])
        else:
            pts = pts[:MAX_POINTS_PER_PILLAR]

        pillar_features.append(pts)
        coords.append([ix, iy])

    return np.stack(pillar_features), np.array(coords)
