# ================================================
# 13. 복수 장애물 간 모든 최소 거리 계산 및 시각화
# ================================================
#
# 목적:
#   이전(12단계)까지는 두 개의 장애물만을 가정하여 최소 거리를 계산했다.
#   하지만 실제 환경에서는 다수의 물체가 동시에 존재할 수 있기 때문에,
#   이번 단계에서는 모든 다각형(클러스터) 쌍을 순회하며 각각의 최소 거리를 계산한다.
#
# 배경:
#   현실적인 주행 환경에서는 차량, 벽, 기둥 등 여러 물체가 동시에 감지된다.
#   따라서 각 물체 간 간격을 전부 파악할 수 있어야 하지만,
#   지나친 조합 계산은 불필요한 시각적 복잡도를 초래할 수 있다.
#   이번 단계에서는 “가능한 모든 쌍의 거리 계산”을 통해
#   DBSCAN의 파라미터 설정과 알파쉐이프 결과가 실제 환경에서
#   어떤 식으로 동작하는지 관찰하는 데 목적이 있다.
#
# 관찰 결과:
#   실제로는 DBSCAN의 eps, min_samples 값이 엉성할 경우
#   노이즈 점까지 클러스터로 포함되어 불필요한 다각형이 생겼고,
#   그 결과 선분이 과도하게 많이 표시되었다.
#   이 과정을 통해 단순히 “모든 물체 간 거리”를 구하는 것이 아니라,
#   “필요한 객체만을 적정 조건으로 구분”하는 기준이 필요함을 깨닫게 되었다.
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / OpenCV 4.9 / shapely 2.0.4 / scikit-learn 1.5
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/13_all_pairs_min_distance.py
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
import itertools

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

depth_np = prediction["depth"].squeeze().detach().cpu().numpy()
h, w = depth_np.shape

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
d_flat = depth_np.flatten().astype(np.float32)

total = u_flat.size
num_points = int(total * 0.15)
idx = np.random.choice(total, num_points, replace=False)

u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]
mask = (d_s <= 30) & (v_s >= h / 2)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
u_s, v_s = u_s[z_mask], v_s[z_mask]

points = np.vstack([cam_right, cam_forward, cam_up]).T

# ------------------------------
# 3단계: DBSCAN + Alpha Shape
# ------------------------------
mask = (
    (points[:,0] >= -10) & (points[:,0] <= 10) &
    (points[:,1] >= 0) & (points[:,1] <= 30)
)
points, u_s, v_s = points[mask], u_s[mask], v_s[mask]

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
# 4단계: 모든 다각형 쌍 최소 거리 계산
# ------------------------------
orig_img = cv2.imread("data/test.jpg")
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

poly_keys = list(polygons.keys())
for (i, j) in itertools.combinations(poly_keys, 2):
    polyA, polyB = polygons[i], polygons[j]
    min_dist = polyA.distance(polyB)
    pA, pB = nearest_points(polyA, polyB)
    pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

    idxA = np.argmin(np.linalg.norm(points[:,:2] - pA, axis=1))
    idxB = np.argmin(np.linalg.norm(points[:,:2] - pB, axis=1))
    uA, vA = int(u_s[idxA]), int(v_s[idxA])
    uB, vB = int(u_s[idxB]), int(v_s[idxB])

    # 선분 및 거리 표시
    cv2.circle(orig_img, (uA, vA), 6, (0,255,0), -1)
    cv2.circle(orig_img, (uB, vB), 6, (0,255,0), -1)
    cv2.line(orig_img, (uA, vA), (uB, vB), (0,255,0), 2)

    mid_u, mid_v = (uA+uB)//2, (vA+vB)//2
    cv2.putText(orig_img, f"{min_dist:.2f} m", (mid_u, mid_v-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

# ------------------------------
# 5단계: 결과 표시
# ------------------------------
plt.figure(figsize=(12,6))
plt.imshow(orig_img)
plt.title("Original Image with All Min-Distance Pairs (Multiple Objects)")
plt.axis("off")
plt.show()
