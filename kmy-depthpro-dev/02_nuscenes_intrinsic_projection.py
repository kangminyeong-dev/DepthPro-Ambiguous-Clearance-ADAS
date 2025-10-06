# nuScenes mini dataset 기반 실험 코드
# 목적: Depth Pro의 예측 결과를 실제 카메라 내부 파라미터(K)를 이용해
#       픽셀 좌표(u,v) → 실제 3D 카메라좌표(X,Y,Z)로 변환하고 시각화한다.
# 사용 데이터: nuScenes CAM_FRONT 이미지 (1600x900)
# 특징:
#   - calibrated_sensor.json의 fx, fy, cx, cy 적용
#   - 깊이 30m 이하 & 이미지 하단 절반만 시각화
#   - 1% 샘플링으로 경량화
#   - 좌표계: X=right, Y=up, Z=forward
#   - 실제 거리 단위 기반으로 깊이 해석 가능
# 

# 터미널 실행 명령어
# python kmy-depthpro-dev/02_nuscenes_intrinsic_projection.py

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
    [1266.417203046554,    0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0,    0.0,    1.0]
])

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat = us.flatten().astype(np.float32)
v_flat = vs.flatten().astype(np.float32)
d_flat = depth.flatten().astype(np.float32)

total = u_flat.size
num_points = int(total * 0.01)
idx = np.random.choice(total, num_points, replace=False)

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
plt.show()
