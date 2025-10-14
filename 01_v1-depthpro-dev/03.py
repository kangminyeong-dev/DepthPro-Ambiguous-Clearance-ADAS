# python 01_v1-depthpro-dev/01_point_y_distribution_combined.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------
# 입력 경로 설정
# ------------------------------
result_dir = "result"
img_path = os.path.join(result_dir, "test.jpg")
depth_path = os.path.join(result_dir, "test.npz")

# ------------------------------
# 이미지 및 깊이 로드
# ------------------------------
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
data = np.load(depth_path)
depth = data["depth"]

# ------------------------------
# 카메라 내부 파라미터 (intrinsic)
# ------------------------------
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# ------------------------------
# 픽셀 → 3D 좌표 변환
# ------------------------------
vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat = us.flatten().astype(np.float32)
v_flat = vs.flatten().astype(np.float32)
d_flat = depth.flatten().astype(np.float32)

num_points = int(u_flat.size * 0.05)
idx = np.random.choice(u_flat.size, num_points, replace=False)

u_s = u_flat[idx]
v_s = v_flat[idx]
d_s = d_flat[idx]

# 25m 이내 + 하단 절반
mask = (d_s <= 25) & (v_s >= h / 2)
u_s = u_s[mask]
v_s = v_s[mask]
d_s = d_s[mask]

x_n = (u_s - cx) / fx
y_n = (v_s - cy) / fy

Xc = x_n * d_s
Yc = y_n * d_s    # 그대로 유지 (도로는 -쪽)
Zc = d_s

# ------------------------------
# Figure: 위(3D 뷰) + 아래(히스토그램)
# ------------------------------
fig = plt.figure(figsize=(10, 10))

# 3D 포인트 클라우드 (위쪽)
ax1 = fig.add_subplot(211, projection="3d")
sc = ax1.scatter(
    Xc, Zc, -Yc,  # 시각화용 (위쪽 양수)
    c=Zc,
    cmap="plasma",
    s=1,
    vmin=0,
    vmax=25
)
ax1.set_xlabel("Camera X (m, right)")
ax1.set_ylabel("Depth Z (m, forward)")
ax1.set_zlabel("Camera Y (m, up)")
ax1.set_xlim(-10, 10)
ax1.set_ylim(0, 25)
ax1.set_zlim(-3, 3)
plt.colorbar(sc, ax=ax1, fraction=0.02, pad=0.1, label="Depth value (0~25 m)")

# 높이(Y) 분포 히스토그램 (아래쪽)
ax2 = fig.add_subplot(212)
ax2.hist(-Yc, bins=100, color="tomato", edgecolor="black", alpha=0.8)
ax2.set_xlabel("Y (m) [camera=0, road is negative]")
ax2.set_ylabel("Frequency")
ax2.set_title("Y-axis Distribution of 3D Points (Depth ≤ 25 m)")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
