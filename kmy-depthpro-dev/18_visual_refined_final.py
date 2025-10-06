# ================================================
# 18. 예외 처리 강화 + 색상 체계 명확화 + 시각화 완성도 개선
# ================================================
#
# 목적:
#   빈 프레임 예외 처리 강화, 거리 기반 색상 체계 명확화, 시각적 일관성 확보를 통해
#   실험 결과를 안정적이고 일관되게 기록하고, 이후 로그·프레임별 대조를 쉽게 만든다.
#
# 배경:
#   이전 버전은 포인트가 없거나 다각형(polygons)이 없을 때 단순히 continue로 스킵했기 때문에
#   추론 중단이나 결과 누락이 자주 발생했다. 이번 버전은 원본 이미지를 그대로 저장하도록 수정하여
#   전체 프레임 시퀀스가 유지되며, 재검증 시 누락된 인덱스 없이 분석이 가능하다.
#
# 주요 변경점:
#   ① 예외 처리 강화 — 포인트/다각형 없음 시 원본 프레임을 CAM_RESULT에 그대로 저장  
#   ② 클러스터 선택 체계 정돈 — 가까운 3개 다각형만 선택, 인덱스 0·1·2 재매핑  
#   ③ 색상 체계 고정 — BLUE(가까움), YELLOW(중간), BROWN(멀음)으로 명확히 고정  
#   ④ 시각화 개선 — Lawngreen/Lightgreen/Green 거리선 + Outline 텍스트로 가독성 향상  
#   ⑤ 시각적 계층 순서 — 선(Line) → 점(Point) → 텍스트(Text) 순서로 표시해 명확한 시인성 확보
#
# 색상 체계 요약:
#       * Blue (가까운 객체), Yellow (중간), Brown (먼 객체)
#       * Lawngreen / Lightgreen / Green (거리 연결선)
#   - Outline 텍스트로 거리 값의 가독성 강화
#
# 실행 환경:
#   Python 3.9.23 / PyTorch 2.4 / OpenCV 4.9 / Shapely 2.0.4  
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/18_visual_refined_final.py
# ================================================

import os
import time
import depth_pro
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import alphashape
from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points
import cv2
import itertools

# ------------------------------
# 0단계: 모델 초기화
# ------------------------------
model, transform = depth_pro.create_model_and_transforms()
model = model.to("cuda").half()
model.eval()

# 카메라 파라미터
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# 경로 설정
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
    # 2단계: Depth 추론
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
    idx = np.random.choice(u_flat.size, int(u_flat.size * 0.1), replace=False)

    u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]
    mask = (d_s <= 30) & (v_s >= h / 2)
    u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

    x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
    cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s
    z_mask = (cam_up >= -0.5) & (cam_up <= 2.0)
    cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
    points = np.vstack([cam_right, cam_forward, cam_up]).T

    # ------------------------------
    # 4단계: DBSCAN + Alpha Shape
    # ------------------------------
    mask = (
        (points[:, 0] >= -5.5) & (points[:, 0] <= 5.5) &
        (points[:, 1] >= 0) & (points[:, 1] <= 20)
    )
    points = points[mask]
    if len(points) == 0:
        print(f"{fname}: 포인트 없음 → 원본 저장")
        orig_img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_dir, fname), orig_img)
        continue

    db = DBSCAN(eps=0.5, min_samples=200).fit(points[:, :2])
    labels = db.labels_
    unique_labels = set(labels)
    polygons, cluster_sizes = {}, {}

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label, :2]
        if len(cluster_points) < 100:
            continue
        shape = alphashape.alphashape(cluster_points, alpha=0.01)
        if isinstance(shape, Polygon):
            polygons[label] = shape
            cluster_sizes[label] = len(cluster_points)

    if len(polygons) == 0:
        print(f"{fname}: 다각형 없음 → 원본 저장")
        orig_img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_dir, fname), orig_img)
        continue

    # ------------------------------
    # 5단계: 가까운 3개 다각형 선택 + 색상 고정
    # ------------------------------
    poly_with_dist = []
    for label, poly in polygons.items():
        centroid = np.mean(np.array(poly.exterior.coords), axis=0)
        dist = np.linalg.norm(centroid)
        poly_with_dist.append((label, poly, dist))

    poly_with_dist.sort(key=lambda x: x[2])
    poly_with_dist = poly_with_dist[:3]

    color_map_ordered = [
        (0, 0, 255),     # Blue (가까운)
        (255, 255, 0),   # Yellow (중간)
        (139, 69, 19)    # Brown (먼)
    ]
    polygons = {}
    color_map = {}
    for idx, (label, poly, _) in enumerate(poly_with_dist):
        polygons[idx] = poly
        color_map[idx] = color_map_ordered[idx]

    line_colors = [
        (124, 252, 0),   # Lawngreen
        (144, 238, 144), # Lightgreen
        (0, 255, 0)      # Green
    ]

    # ------------------------------
    # 6단계: 최소 거리 계산 + 시각화
    # ------------------------------
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    poly_keys = list(polygons.keys())
    line_data = []

    for idx, (i, j) in enumerate(itertools.combinations(poly_keys, 2)):
        polyA, polyB = polygons[i], polygons[j]
        min_dist = polyA.distance(polyB)
        pA, pB = nearest_points(polyA, polyB)
        pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])
        line = LineString([pA, pB])

        # 다른 다각형 관통 방지
        if any(line.intersects(c) for k, c in polygons.items() if k not in (i, j)):
            continue
        if not (1.0 <= min_dist <= 7.0):
            continue

        line_color = line_colors[idx % len(line_colors)]
        uA, vA = int(pA[0]), int(pA[1])
        uB, vB = int(pB[0]), int(pB[1])

        # 선→점→텍스트 순서
        cv2.line(orig_img, (uA, vA), (uB, vB), line_color, 2)
        cv2.circle(orig_img, (uA, vA), 10, color_map[i], -1)
        cv2.circle(orig_img, (uB, vB), 10, color_map[j], -1)

        mid_u, mid_v = (uA + uB)//2, (vA + vB)//2
        text = f"{min_dist:.2f} m"
        cv2.putText(orig_img, text, (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(orig_img, text, (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, line_color, 2, cv2.LINE_AA)
        line_data.append((pA, pB, min_dist))

    # ------------------------------
    # 7단계: 시각화 저장
    # ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image with Min-Distance")
    axes[0].axis("off")

    for label, poly in polygons.items():
        x, y = poly.exterior.xy
        axes[1].plot(x, y, 'r-', linewidth=1.5)
        centroid = np.mean(np.array(poly.exterior.coords), axis=0)
        axes[1].text(centroid[0], centroid[1],
                     f"{cluster_sizes.get(label,'?')} pts", color='green', fontsize=9, ha='center')

    for (pA, pB, dist) in line_data:
        axes[1].plot([pA[0], pB[0]], [pA[1], pB[1]], 'g--', linewidth=2)
        axes[1].scatter([pA[0], pB[0]], [pA[1], pB[1]], c='green', s=50, marker='x')
        mid_x, mid_y = (pA[0]+pB[0])/2, (pA[1]+pB[1])/2
        axes[1].text(mid_x, mid_y, f"{dist:.2f} m", color='green', fontsize=9)

    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(0, 30)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Camera X (m, right)")
    axes[1].set_ylabel("Camera Y (m, forward)")
    axes[1].set_title("Top-View Clustering & Min Distance")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"{fname} 처리 완료 - {elapsed:.2f}초")

# ------------------------------
# 8단계: 평균 처리 시간 출력
# ------------------------------
if len(times) > 0:
    print(f"\n평균 처리 시간: {np.mean(times):.2f}초/장, 총 {len(times)}장 처리됨")
else:
    print("처리된 이미지가 없습니다.")
