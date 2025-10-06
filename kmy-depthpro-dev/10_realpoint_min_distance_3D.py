# ================================================
# 10. 실제 3D 포인트 기반 최소거리 확장 (AlphaShape + Depth)
# ================================================
#
# 목적:
#   09단계에서는 탑뷰(XY 평면) 기준으로만 다각형 최소거리를 계산했다.
#   이때는 물체 간의 "투영 거리"만 구할 수 있었으며, 높이(Z축) 정보는 포함되지 않았다.
#
#   이번 10단계에서는 Alpha Shape에서 얻은 최근접 두 점(pA, pB)을
#   실제 포인트 클라우드에서 다시 역추적하여,
#   각 점의 (X, Y, Z) 좌표를 가져오고,
#   이 두 점 사이의 진짜 3D 공간 거리(Euclidean distance)를 계산한다.
#
# 핵심 비교:
#   09단계 → 2D 투영 거리
#      두 장애물 사이 최소 거리: 4.436 m
#      최근접점 A (X, Y): (3.048, 8.377)
#      최근접점 B (X, Y): (-0.200, 11.398)
#
#   10단계 → 3D 공간 거리
#      탑뷰 최소거리(2D): 4.429 m
#      실제 3D 거리:      4.439 m
#      포인트 A (X,Y,Z):  [ 3.0255,  8.3708, -0.7237]
#      포인트 B (X,Y,Z):  [-0.2005, 11.4049, -0.4277]
#
#   두 결과의 차이는 (약 0.03m 미만) Z축 높이 차이가 존재하기 때문이며,
#   이 실험은 향후 실제 Clearance Detection 시 
#   "3D 실거리 기반 판단"의 정확도를 검증하기 위한 준비 단계이다.
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / shapely 2.0.4
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/10_realpoint_min_distance_3D.py
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

depth = prediction["depth"]
depth_np = depth.squeeze().detach().cpu().numpy()
h, w = depth_np.shape

# ------------------------------
# 2단계: 픽셀 → 3D 포인트 변환
# ------------------------------
K = np.array([
    [1266.417203046554,    0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0,    0.0,    1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat, v_flat = us.flatten().astype(np.float32), vs.flatten().astype(np.float32)
d_flat = depth_np.flatten().astype(np.float32)

# 샘플링 및 필터링
total = u_flat.size
num_points = int(total * 0.1)
idx = np.random.choice(total, num_points, replace=False)
u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]

mask = (d_s <= 30) & (v_s >= h / 2)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

# 정규화 → 카메라 좌표계
x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
points = np.vstack([cam_right, cam_forward, cam_up]).T  # (N,3)

# ------------------------------
# 3단계: DBSCAN + Alpha Shape (XY 평면 기준)
# ------------------------------
mask = (
    (points[:,0] >= -30) & (points[:,0] <= 30) &
    (points[:,1] >= 0) & (points[:,1] <= 30)
)
points = points[mask]

db = DBSCAN(eps=0.08, min_samples=30).fit(points[:, :2])
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
# 4단계: 최소 거리 계산 (3D 반영)
# ------------------------------
plt.figure(figsize=(8, 8))

if len(polygons) >= 2:
    keys = list(polygons.keys())
    polyA, polyB = polygons[keys[0]], polygons[keys[1]]

    # 2D 최소 거리 계산
    min_dist_2d = polyA.distance(polyB)
    pA, pB = nearest_points(polyA, polyB)
    pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

    # 3D 실제 포인트 찾기
    idxA = np.argmin(np.linalg.norm(points[:, :2] - pA, axis=1))
    idxB = np.argmin(np.linalg.norm(points[:, :2] - pB, axis=1))
    realA, realB = points[idxA], points[idxB]

    # 3D 거리 계산
    min_dist_3d = np.linalg.norm(realA - realB)

    print(f"탑뷰 최소거리(2D): {min_dist_2d:.3f} m")
    print(f"실제 3D 거리: {min_dist_3d:.3f} m")
    print("포인트 A (X,Y,Z):", realA)
    print("포인트 B (X,Y,Z):", realB)

    # ------------------------------
    # 5단계: 시각화
    # ------------------------------
    for label in polygons:
        cluster_points = points[labels == label, :2]
        plt.scatter(cluster_points[:,0], cluster_points[:,1], s=2, alpha=0.5)
        x, y = polygons[label].exterior.xy
        plt.plot(x, y, 'r-', linewidth=1.5)

    # 최소 거리 선분 표시 (실제 3D 대응점 기준)
    plt.plot([realA[0], realB[0]], [realA[1], realB[1]], 'g--', linewidth=2)
    plt.scatter([realA[0], realB[0]], [realA[1], realB[1]], c='green', s=50, marker='x')

plt.xlabel("Camera X (m, right)")
plt.ylabel("Camera Y (m, forward)")
plt.title("Top-View Clustering + 3D-Aware Min Distance")
plt.xlim(-10, 10)
plt.ylim(0, 20)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
