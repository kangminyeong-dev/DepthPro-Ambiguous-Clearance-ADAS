# python 01_v1-depthpro-dev/01.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

result_dir = "result"
img_path = os.path.join(result_dir, "test.jpg")
depth_path = os.path.join(result_dir, "test.npz")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
data = np.load(depth_path)
depth = data["depth"]

K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
 
vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat = us.flatten().astype(np.float32)
v_flat = vs.flatten().astype(np.float32)
d_flat = depth.flatten().astype(np.float32)

num_points = int(u_flat.size * 0.05)
idx = np.random.choice(u_flat.size, num_points, replace=False)

u_s = u_flat[idx]
v_s = v_flat[idx]
d_s = d_flat[idx]

mask = (d_s <= 30) & (v_s >= h / 2)
u_s = u_s[mask]
v_s = v_s[mask]
d_s = d_s[mask]

x_n = (u_s - cx) / fx
y_n = (v_s - cy) / fy

Xc = x_n * d_s
Yc = y_n * d_s
Zc = d_s

points = np.stack([Xc, Yc, Zc], axis=1)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(Xc, Zc, -Yc, c=Zc, cmap="plasma", s=1, vmin=0, vmax=30)

ax.set_xlabel("Camera X (m, right)")
ax.set_ylabel("Depth Z (m, forward)")
ax.set_zlabel("Camera Y (m, up)")

ax.set_xlim(-10,10)
ax.set_ylim(0,15)
ax.set_zlim(-3,3)

plt.colorbar(sc, label="Depth value (0~30 m)")
plt.tight_layout()
plt.show(block=False)

"""
"""

grid_size = 0.2
x_min, x_max = -10, 10
z_min, z_max = 0, 30
nx = int((x_max - x_min) / grid_size)
nz = int((z_max - z_min) / grid_size)

ix = np.clip(((points[:, 0] - x_min) / grid_size).astype(int), 0, nx - 1)
iz = np.clip(((points[:, 2] - z_min) / grid_size).astype(int), 0, nz - 1)

cell_dict = {}
for i, (cx, cz) in enumerate(zip(ix, iz)):
    key = (cx, cz)
    if key not in cell_dict:
        cell_dict[key] = []
    cell_dict[key].append(i)

density_th = 100
height_std_th = 0.15
min_points_cell = 10

ground_mask = np.zeros(len(points), dtype=bool)
topview_map = np.zeros((nz, nx), dtype=np.uint8)

for key, idxs in cell_dict.items():
    cz, cx = key[1], key[0]
    if len(idxs) < min_points_cell:
        ground_mask[idxs] = True
        topview_map[cz, cx] = 1
        continue

    local_y = points[idxs, 1]
    height_std = np.std(local_y)
    density = len(idxs)

    if (density < density_th) and (height_std < height_std_th):
        ground_mask[idxs] = True
        topview_map[cz, cx] = 1

non_ground_pts = points[~ground_mask]
ground_pts = points[ground_mask]
print(f"Total points: {len(points)}, Ground removed: {np.sum(ground_mask)}")

# Xg, Zg = np.meshgrid(
#     np.linspace(x_min, x_max, nx),
#     np.linspace(z_min, z_max, nz)
# )
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")

# surf = ax.plot_surface(Xg, Zg, np.nan_to_num(topview_map, nan=0),
#                        cmap="coolwarm", alpha=0.8, linewidth=0, antialiased=False)

# ax.scatter(non_ground_pts[:, 0], non_ground_pts[:, 2], -non_ground_pts[:, 1],
#            c=-non_ground_pts[:, 1], cmap="plasma", s=2)

# ax.set_xlabel("Camera X (m, right)")
# ax.set_ylabel("Depth Z (m, forward)")
# ax.set_zlabel("Camera Y (m, up)")

# ax.set_xlim(-10, 10)
# ax.set_ylim(0, 15)
# ax.set_zlim(-3, 3)
# ax.view_init(elev=25, azim=-60)

# plt.colorbar(surf, label="Average height (m)")
# plt.tight_layout()
# plt.show(block=False)

"""
"""

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

Xv = non_ground_pts[:, 0]
Yv = -non_ground_pts[:, 1]
Zv = non_ground_pts[:, 2]

sc = ax.scatter(Xv, Zv, Yv,
                c=Yv, cmap="plasma", s=2, vmin=-3, vmax=3)

ax.set_xlabel("Camera X (m, right)")
ax.set_ylabel("Depth Z (m, forward)")
ax.set_zlabel("Camera Y (m, up)")

ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_zlim(-3, 3)

plt.colorbar(sc, label="Camera Y (height, m)")
plt.tight_layout()
plt.show()