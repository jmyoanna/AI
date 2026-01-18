# data/generate_fake_lidar.py
import numpy as np

N = 10000

points = np.random.uniform(
    low=[-10, -10, -2, 0],
    high=[10, 10, 2, 1],
    size=(N, 4)  # x, y, z, intensity
)

np.save("data/fake_lidar.npy", points)
print("Saved fake_lidar.npy")
