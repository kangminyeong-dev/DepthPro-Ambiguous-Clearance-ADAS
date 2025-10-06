# ================================================
# 12. 실제 최소거리(m 단위)를 원본 이미지에 시각적으로 표시
# ================================================
#
# 목적:
#   Depth Pro 추론 결과로 얻은 3D 포인트 기반 두 장애물 간 최소 거리를
#   원본 이미지 위에 "수치로 직접 표시"하는 단계.
#   이전 단계(11)에서는 점과 선만 그려졌지만,
#   이번엔 실제 미터 단위 거리 값을 영상 중앙에 텍스트로 출력한다.
#
# 배경:
#   이 시점의 핵심은 “시각적으로 명확하게 보여주는 결과물”이었다.
#   단순히 측정값만 터미널에 출력하는 것이 아니라,
#   사람이 봐도 직관적으로 거리 감각을 이해할 수 있게끔
#   원본 프레임 위에 선분과 거리값을 함께 오버레이했다.
#
# 기술 포인트:
#   ① DBSCAN + Alpha Shape으로 장애물 클러스터 경계 추출
#   ② shapely.nearest_points()로 두 다각형 사이 최단 거리 계산
#   ③ 실제 깊이 기반 3D 포인트에서 최근접 점 매칭 (XY→3D)
#   ④ 해당 픽셀 인덱스(u,v)를 사용해 원본 이미지에 선분 매핑
#   ⑤ 중심 좌표(mid_u, mid_v)에 거리값 표시 (OpenCV putText)
#
# 결과:
#   원본 프레임 위에 두 장애물 간 최소 거리(단위 m)가 녹색 텍스트로 출력된다.
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / OpenCV 4.9 / shapely 2.0.4
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/12_visualize_real_distance_overlay.py
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
# 1단계: GPU 추론 (Depth 생성)
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
# 2단계: 픽셀 → 3D 변환
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
# 4단계: 최소 거리 계산 + 원본 프레임에 시각화
# ------------------------------
if len(polygons) >= 2:
    keys = list(polygons.keys())
    polyA, polyB = polygons[keys[0]], polygons[keys[1]]
    min_dist = polyA.distance(polyB)
    pA, pB = nearest_points(polyA, polyB)
    pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

    idxA = np.argmin(np.linalg.norm(points[:,:2] - pA, axis=1))
    idxB = np.argmin(np.linalg.norm(points[:,:2] - pB, axis=1))
    realA, realB = points[idxA], points[idxB]

    print(f"두 장애물 최소 거리: {min_dist:.3f} m")
    print("A 포인트 (X,Y,Z):", realA)
    print("B 포인트 (X,Y,Z):", realB)

    # ------------------------------
    # 5단계: 원본 이미지에 점, 선, 거리값 표시
    # ------------------------------
    orig_img = cv2.imread("data/test.jpg")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    uA, vA = int(u_s[idxA]), int(v_s[idxA])
    uB, vB = int(u_s[idxB]), int(v_s[idxB])

    # 두 점 및 연결선
    cv2.circle(orig_img, (uA, vA), 8, (0,255,0), -1)
    cv2.circle(orig_img, (uB, vB), 8, (0,255,0), -1)
    cv2.line(orig_img, (uA, vA), (uB, vB), (0,255,0), 3)

    # 중간 위치 텍스트 표시
    mid_u, mid_v = (uA + uB)//2, (vA + vB)//2
    cv2.putText(orig_img, f"{min_dist:.2f} m", (mid_u, mid_v-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

    # 표시
    plt.figure(figsize=(12,6))
    plt.imshow(orig_img)
    plt.title("Original Image with Real Distance Overlay")
    plt.axis("off")
    plt.show()
