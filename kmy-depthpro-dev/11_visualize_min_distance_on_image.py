# ================================================
# 11. 3D 최소거리 결과를 원본 이미지에 재투영 (Back-Projection Verification)
# ================================================
#
# 목적:
#   탑뷰(Top-View)에서 계산된 두 물체 간 최소 거리와 해당 최근접점 정보를
#   다시 원본 RGB 이미지 위에 표시함으로써,
#   Depth-to-Image 매핑의 일관성과 좌표 정합성을 검증한다.
#
# 배경 고민:
#   처음에는 “탑뷰 좌표로 계산된 선분을 원본 이미지에 어떻게 다시 매핑할까?”
#   하는 의문이 있었다.
#   하지만 곰곰이 생각해보니, 깊이 추론과 왜곡 보정을 거쳤더라도
#   각 픽셀(u,v)은 고유 인덱스를 갖고 있으므로,
#   그 인덱스를 통해 다시 원본 이미지 위에 대응점을 찍을 수 있었다.
#   결국 이 접근으로 간단하게 선분을 복원할 수 있었다.
#
# 핵심 흐름:
#   Depth → 3D 변환 → 클러스터링(DBSCAN+Alpha-Shape) → 최소거리 계산
#   → 최근접점 인덱스 역추적 → 원본 이미지 픽셀 좌표로 매핑 → 시각화
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / OpenCV 4.9 / shapely 2.0.4
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/11_visualize_min_distance_on_image.py
# ================================================

from PIL import Image
import depth_pro
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import alphashape
from shapely.geometry import Polygon
from shapely.ops import nearest_points
import cv2

# ------------------------------
# 1단계: GPU 추론
# ------------------------------
model, transform = depth_pro.create_model_and_transforms()
model = model.to("cuda").half()
model.eval()

image, _, f_px = depth_pro.load_rgb("data/test.jpg")
image = transform(image).to("cuda").half()

with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

depth = prediction["depth"].squeeze().detach().cpu().numpy()
h, w = depth.shape

# ------------------------------
# 2단계: 포인트 클라우드 변환
# ------------------------------
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat, v_flat = us.flatten().astype(np.float32), vs.flatten().astype(np.float32)
d_flat = depth.flatten().astype(np.float32)

total = u_flat.size
num_points = int(total * 0.1)
idx = np.random.choice(total, num_points, replace=False)

u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]
mask = (d_s <= 30) & (v_s >= h / 2)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
u_s, v_s, d_s = u_s[z_mask], v_s[z_mask], d_s[z_mask]

points = np.vstack([cam_right, cam_forward, cam_up]).T

# ------------------------------
# 3단계: DBSCAN + Alpha Shape
# ------------------------------
mask = (
    (points[:,0] >= -30) & (points[:,0] <= 30) &
    (points[:,1] >= 0) & (points[:,1] <= 30)
)
points = points[mask]
u_s, v_s = u_s[mask], v_s[mask]

db = DBSCAN(eps=0.08, min_samples=30).fit(points[:,:2])
labels = db.labels_
unique_labels = set(labels)

polygons = {}
for label in unique_labels:
    if label == -1:
        continue
    cluster_points = points[labels == label, :2]
    if len(cluster_points) < 100:
        continue
    shape = alphashape.alphashape(cluster_points, alpha=0.05)
    if isinstance(shape, Polygon):
        polygons[label] = shape

# ------------------------------
# 4단계: 최소거리 계산
# ------------------------------
keys = list(polygons.keys())
polyA, polyB = polygons[keys[0]], polygons[keys[1]]
min_dist = polyA.distance(polyB)
pA, pB = nearest_points(polyA, polyB)
pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

# 실제 포인트 인덱스 추적
idxA = np.argmin(np.linalg.norm(points[:,:2] - pA, axis=1))
idxB = np.argmin(np.linalg.norm(points[:,:2] - pB, axis=1))
realA, realB = points[idxA], points[idxB]

print(f"두 장애물 최소거리(탑뷰): {min_dist:.3f} m")
print("포인트 A (X,Y,Z):", realA)
print("포인트 B (X,Y,Z):", realB)

# ------------------------------
# 5단계: 시각화 (탑뷰 + 원본 이미지 병합)
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# (좌) 탑뷰 시각화
ax1 = axes[0]
for label in polygons:
    cluster_points = points[labels == label, :2]
    ax1.scatter(cluster_points[:,0], cluster_points[:,1], s=2, alpha=0.5)
    x, y = polygons[label].exterior.xy
    ax1.plot(x, y, 'r-', linewidth=1.5)
ax1.plot([realA[0], realB[0]], [realA[1], realB[1]], 'g--', linewidth=2)
ax1.scatter([realA[0], realB[0]], [realA[1], realB[1]], c='green', s=50, marker='x')
ax1.set_xlabel("Camera X (m, right)")
ax1.set_ylabel("Camera Y (m, forward)")
ax1.set_xlim(-10, 10)
ax1.set_ylim(0, 20)
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("Top-View Clustering & Min Distance")

# (우) 원본 이미지 시각화
ax2 = axes[1]
orig_img = cv2.imread("data/test.jpg")
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
uA, vA = int(u_s[idxA]), int(v_s[idxA])
uB, vB = int(u_s[idxB]), int(v_s[idxB])
cv2.circle(orig_img, (uA, vA), 8, (255,0,0), -1)
cv2.circle(orig_img, (uB, vB), 8, (0,255,0), -1)
cv2.line(orig_img, (uA, vA), (uB, vB), (0,255,255), 3)
ax2.imshow(orig_img)
ax2.axis("off")
ax2.set_title("Original Image with Min-Distance Mapping")

plt.tight_layout()
plt.show()

