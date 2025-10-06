# ================================================
# 16. 조건 필터링 실험 및 클러스터 크기 기반 검증 (Condition Filtering with Cluster Size Insight)
# ================================================
#
# 목적:
#   15단계에서 확립한 DBSCAN-AlphaShape 파라미터를 기반으로,
#   실제 주행 환경에서 의미 있는 객체 간 거리만 남기기 위한
#   필터링 로직을 적용하고 검증한다.
#
# 배경:
#   실도로 환경은 평평하지 않으며, 노면 경사나 그림자, 낮은 구조물로 인해
#   불필요한 클러스터가 생성되는 현상이 있었다.
#   따라서 z축 필터를 (-0.5~2.0m)로 좁혀 지면 잡음을 최소화하고,
#   DBSCAN 밀도 기준을 강화하여 의미 있는 물체만 검출하도록 했다.
#
# 핵심 포인트:
#   - 단순히 시각적 결과에 의존하지 않고,
#     각 클러스터별 포인트 개수를 직접 출력·표시함으로써
#     파라미터 튜닝의 근거를 수치적으로 확보했다.
#   - 이 과정을 통해 eps·min_samples·alpha 조정 시
#     ‘객체별 점 밀도 분포’를 근거로 판단할 수 있었다.
#
# 주요 조정 사항:
#   - z축 범위: -0.55~3.0m → -0.5~2.0m (노면 필터 강화)
#   - DBSCAN: eps=0.25 → 0.6, min_samples=25 → 200 (확실한 밀집군만 유지)
#   - 탐색 범위: X축 -5.5~5.5m, Y축 0~20m (실제 차량 전방 시야에 맞춤)
#   - 거리 조건: 1.0~7.0m 사이의 최소 거리만 시각화 (비현실적 거리 제거)
#   - 클러스터 포인트 수를 시각적으로 표시하여 튜닝 근거 확보
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / OpenCV 4.9 / shapely 2.0.4
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/16_condition_filtering_cluster_size.py
# ================================================


import os
import time
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
# 0단계: 모델 초기화
# ------------------------------
model, transform = depth_pro.create_model_and_transforms()
model = model.to("cuda").half()
model.eval()

# 카메라 내외부 파라미터
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# 입력/출력 경로
input_dir = "CAM_FRONT"
output_dir = "CAM_RESULT"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 1단계: 이미지 순차 처리
# ------------------------------
times = []
for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    start_time = time.time()
    img_path = os.path.join(input_dir, fname)

    # ------------------------------
    # 2단계: Depth Pro 추론
    # ------------------------------
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(image).to("cuda").half()
    with torch.no_grad():
        prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"].squeeze().detach().cpu().numpy()
    h, w = depth.shape

    # ------------------------------
    # 3단계: 포인트클라우드 변환
    # ------------------------------
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

    # 지면 필터 강화 (-0.5 ~ 2.0m)
    z_mask = (cam_up >= -0.5) & (cam_up <= 2.0)
    cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
    u_s, v_s, d_s = u_s[z_mask], v_s[z_mask], d_s[z_mask]
    points = np.vstack([cam_right, cam_forward, cam_up]).T

    # ------------------------------
    # 4단계: DBSCAN + Alpha Shape
    # ------------------------------
    mask = (
        (points[:, 0] >= -5.5) & (points[:, 0] <= 5.5) &
        (points[:, 1] >= 0) & (points[:, 1] <= 20)
    )
    points, u_s, v_s = points[mask], u_s[mask], v_s[mask]

    if len(points) == 0:
        print(f"{fname}: 포인트 없음, 스킵")
        continue

    db = DBSCAN(eps=0.6, min_samples=200).fit(points[:, :2])
    labels = db.labels_
    unique_labels = set(labels)

    polygons = {}
    cluster_sizes = {}
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label, :2]
        if len(cluster_points) < 100:
            continue
        shape = alphashape.alphashape(cluster_points, alpha=0.05)
        if isinstance(shape, Polygon):
            polygons[label] = shape
            cluster_sizes[label] = len(cluster_points)

    # ------------------------------
    # 5단계: 최소 거리 계산 + 조건 필터링
    # ------------------------------
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    poly_keys = list(polygons.keys())
    line_data = []
    for (i, j) in itertools.combinations(poly_keys, 2):
        polyA, polyB = polygons[i], polygons[j]
        min_dist = polyA.distance(polyB)
        pA, pB = nearest_points(polyA, polyB)
        pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

        idxA = np.argmin(np.linalg.norm(points[:, :2] - pA, axis=1))
        idxB = np.argmin(np.linalg.norm(points[:, :2] - pB, axis=1))
        uA, vA = int(u_s[idxA]), int(v_s[idxA])
        uB, vB = int(u_s[idxB]), int(v_s[idxB])

        # 거리 조건 필터 (1~7m)
        if not (1.0 <= min_dist <= 7.0):
            continue

        # 원본 시각화
        cv2.circle(orig_img, (uA, vA), 6, (0, 255, 0), -1)
        cv2.circle(orig_img, (uB, vB), 6, (0, 255, 0), -1)
        cv2.line(orig_img, (uA, vA), (uB, vB), (0, 255, 0), 2)
        mid_u, mid_v = (uA + uB) // 2, (vA + vB) // 2
        cv2.putText(orig_img, f"{min_dist:.2f} m", (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        line_data.append((pA, pB, min_dist))

    # ------------------------------
    # 6단계: 원본 + 탑뷰 시각화
    # ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image with Min-Distance (Filtered)")
    axes[0].axis("off")

    axes[1].scatter(points[:, 0], points[:, 1], s=2, alpha=0.5)
    for label in polygons:
        x, y = polygons[label].exterior.xy
        axes[1].plot(x, y, 'r-', linewidth=1.5)
        centroid = np.mean(np.array(polygons[label].exterior.coords), axis=0)
        axes[1].text(centroid[0], centroid[1], f"{cluster_sizes[label]} pts",
                     color='orange', fontsize=9, ha='center')

    for (pA, pB, dist) in line_data:
        axes[1].plot([pA[0], pB[0]], [pA[1], pB[1]], 'g--', linewidth=2)
        axes[1].scatter([pA[0], pB[0]], [pA[1], pB[1]], c='green', s=50, marker='x')
        mid_x, mid_y = (pA[0] + pB[0]) / 2, (pA[1] + pB[1]) / 2
        axes[1].text(mid_x, mid_y, f"{dist:.2f} m", color='green', fontsize=9)

    axes[1].set_xlabel("Camera X (m, right)")
    axes[1].set_ylabel("Camera Y (m, forward)")
    axes[1].set_title("Top-View Clustering & Min Distance (Filtered by Condition)")
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(0, 30)
    axes[1].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"{fname} 처리 완료 - {elapsed:.2f}초")

# ------------------------------
# 7단계: 평균 처리 시간 출력
# ------------------------------
if len(times) > 0:
    print(f"\n평균 처리 시간: {np.mean(times):.2f}초/장, 총 {len(times)}장 처리됨")
else:
    print("처리된 이미지가 없습니다.")
