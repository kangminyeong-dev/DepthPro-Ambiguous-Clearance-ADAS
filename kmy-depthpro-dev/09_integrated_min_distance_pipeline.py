# ================================================
# 09. DepthPro 기반 실전형 탑뷰 최소거리 계산 (통합)
# ================================================
#
# 목적:
#   지금까지 분리 실험했던 08단계 전체를 실제 파이프라인으로
#   하나로 통합하여 GPU 추론부터 거리계산까지 자동 수행한다.
#
# 흐름 요약:
#   1) DepthPro로 깊이 추정 (FP16 GPU)
#   2) 픽셀 → 3D 포인트 변환 (카메라 좌표계)
#   3) 탑뷰 정제 및 필터링
#   4) DBSCAN으로 물체 분리
#   5) Alpha-Shape으로 외곽 윤곽 생성
#   6) Shapely로 두 물체 사이 실제 최소 거리 계산
#
# 학습 포인트:
#   - 이 버전은 08-04~08-06 단계의 핵심 아이디어(Alpha-Shape + nearest_points)
#     를 실제 DepthPro 출력 포인트에 적용한 첫 통합 예시다.
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / shapely 2.0.4 / scikit-learn 1.5
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/09_integrated_min_distance_pipeline.py
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
# 1단계: GPU 추론 (FP16)
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

# 샘플링 (10%)
total = u_flat.size
num_points = int(total * 0.1)
idx = np.random.choice(total, num_points, replace=False)
u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]

# 전방 30m + 하단 영역
mask = (d_s <= 30) & (v_s >= h / 2)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

# 정규화 좌표계 → 카메라 좌표계
x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

# 지면 정제
z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward = cam_right[z_mask], cam_forward[z_mask]
points = np.vstack([cam_right, cam_forward]).T  # (N, 2)

# ------------------------------
# 3단계: DBSCAN + Alpha Shape
# ------------------------------
mask = (
    (points[:, 0] >= -30) & (points[:, 0] <= 30) &
    (points[:, 1] >= 0) & (points[:, 1] <= 30)
)
points = points[mask]

db = DBSCAN(eps=0.08, min_samples=30).fit(points)
labels = db.labels_
unique_labels = set(labels)

polygons = {}
for label in unique_labels:
    if label == -1:
        continue
    cluster_points = points[labels == label]
    if len(cluster_points) < 100:
        continue
    shape = alphashape.alphashape(cluster_points, alpha=0.05)
    if isinstance(shape, Polygon):
        polygons[label] = shape

# ------------------------------
# 4단계: 최소 거리 계산 + 시각화
# ------------------------------
plt.figure(figsize=(8, 8))

if len(polygons) >= 2:
    keys = list(polygons.keys())
    polyA, polyB = polygons[keys[0]], polygons[keys[1]]
    min_dist = polyA.distance(polyB)
    pA, pB = nearest_points(polyA, polyB)
    pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

    # ===== 추가된 출력 =====
    print(f"두 장애물 사이 최소 거리: {min_dist:.3f} m (탑뷰 XY 기준)")
    print(f"최근접점 A (X, Y): ({pA[0]:.3f}, {pA[1]:.3f})")
    print(f"최근접점 B (X, Y): ({pB[0]:.3f}, {pB[1]:.3f})")
    # =======================

    for label in polygons:
        cluster_points = points[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=2, alpha=0.5)
        x, y = polygons[label].exterior.xy
        plt.plot(x, y, 'r-', linewidth=1.5)

    plt.plot([pA[0], pB[0]], [pA[1], pB[1]], 'g--', linewidth=2)
    plt.scatter([pA[0], pB[0]], [pA[1], pB[1]], c='green', s=50, marker='x')

plt.xlabel("Camera X (m, right)")
plt.ylabel("Camera Y (m, forward)")
plt.title("Top-View Clustering & Real Min Distance")
plt.xlim(-10, 10)
plt.ylim(0, 20)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
