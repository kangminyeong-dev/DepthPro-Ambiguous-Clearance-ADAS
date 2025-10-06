# ======================================================
# 19. 지면 필터링 강화 및 시각적 일관성(범례 포함)
# ======================================================
#
# 목적:
#   불필요하게 높은 장애물 포인트를 제거하고,
#   모든 결과 이미지에 색상 체계(Far/Mid/Close)를 명확히 표시하여
#   처음 쓰는 사람도 거리 구간 해석이 쉽게 가능하도록 개선한다.
#
# 핵심 변경점:
#   ① 지면 필터링(cam_up) 범위 강화: [-0.5, 0.0]
#   ② 색상 매핑 구조 변경: 딕셔너리 기반 명시적 인덱스 접근
#   ③ 범례 함수(add_legend) 추가 및 모든 프레임에 표시
#   ④ 파라미터 튜닝 최적화 및 인터페이스 중심 개선
#
# 색상 체계:
#   * Yellow (가까움), Blue (중간), Brown (먼)
#   * Lawngreen / Lightgreen / Green (거리 연결선)
#   - Outline 텍스트로 거리 값의 가독성 강화
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/19_ground_filter_legend_final.py
# ======================================================

import os
import time
import depth_pro
import torch
import numpy as np
import cv2
import itertools
import alphashape
from shapely.geometry import Polygon, LineString
from shapely.ops import nearest_points
from sklearn.cluster import DBSCAN

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
# 범례 함수 정의
# ------------------------------
def add_legend(img):
    legend_items = [("Far", (139, 69, 19)), ("Mid", (0, 0, 255)), ("Close", (255, 255, 0))]
    for idx, (text, color) in enumerate(legend_items):
        y_pos = 40 + idx * 40
        cv2.circle(img, (40, y_pos), 10, color, -1)
        cv2.putText(img, text, (70, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img

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
    # 추론
    # ------------------------------
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(image).to("cuda").half()
    with torch.no_grad():
        prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"].squeeze().detach().cpu().numpy()
    h, w = depth.shape

    # ------------------------------
    # 포인트클라우드 변환
    # ------------------------------
    vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u_flat, v_flat = us.flatten().astype(np.float32), vs.flatten().astype(np.float32)
    d_flat = depth.flatten().astype(np.float32)

    # 샘플링 (10%)
    total = u_flat.size
    num_points = int(total * 0.1)
    idx = np.random.choice(total, num_points, replace=False)
    u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]

    # 범위 제한
    mask = (d_s <= 30) & (v_s >= h / 2)
    u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

    # 카메라 좌표계
    x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
    cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

    # ------------------------------
    # 지면 필터링 강화 (카메라 장착 높이 기준)
    # ------------------------------
    z_mask = (cam_up >= -0.5) & (cam_up <= 0.0)
    cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
    u_s, v_s, d_s = u_s[z_mask], v_s[z_mask], d_s[z_mask]

    points = np.vstack([cam_right, cam_forward, cam_up]).T

    # ------------------------------
    # DBSCAN + Alpha Shape
    # ------------------------------
    mask = ((points[:, 0] >= -5) & (points[:, 0] <= 5) &
            (points[:, 1] >= 0) & (points[:, 1] <= 25))
    points, u_s, v_s = points[mask], u_s[mask], v_s[mask]

    if len(points) == 0:
        print(f"{fname}: 포인트 없음, 원본 저장")
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = add_legend(orig_img)
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        continue

    db = DBSCAN(eps=0.55, min_samples=200).fit(points[:, :2])
    labels = db.labels_
    unique_labels = set(labels)

    polygons = {}
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label, :2]
        if len(cluster_points) < 100:
            continue
        shape = alphashape.alphashape(cluster_points, alpha=0.005)
        if isinstance(shape, Polygon):
            polygons[label] = shape

    if len(polygons) == 0:
        print(f"{fname}: 다각형 없음, 원본 저장")
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = add_legend(orig_img)
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        continue

    # ------------------------------
    # 가까운 3개 다각형만 선택 (원점 기준)
    # ------------------------------
    poly_with_dist = []
    for label, poly in polygons.items():
        centroid = np.mean(np.array(poly.exterior.coords), axis=0)
        dist = np.linalg.norm(centroid)
        poly_with_dist.append((label, poly, dist))
    poly_with_dist.sort(key=lambda x: x[2])
    poly_with_dist = poly_with_dist[:3]
    polygons = {idx: poly for idx, (label, poly, _) in enumerate(poly_with_dist)}

    # ------------------------------
    # 색상 고정 (딕셔너리 기반)
    # ------------------------------
    color_map_ordered = {
        0: (255, 255, 0),   # Yellow (Close)
        1: (0, 0, 255),     # Blue   (Mid)
        2: (139, 69, 19)    # Brown  (Far)
    }
    line_colors = [
        (124, 252, 0),
        (144, 238, 144),
        (0, 255, 0)
    ]

    # ------------------------------
    # 최소 거리 계산 및 시각화
    # ------------------------------
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    poly_keys = list(polygons.keys())
    for idx, (i, j) in enumerate(itertools.combinations(poly_keys, 2)):
        polyA, polyB = polygons[i], polygons[j]
        min_dist = polyA.distance(polyB)
        pA, pB = nearest_points(polyA, polyB)
        pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

        line = LineString([pA, pB])
        skip = False
        for k, polyC in polygons.items():
            if k in (i, j):
                continue
            if line.intersects(polyC):
                skip = True
                break
        if skip or not (1.0 <= min_dist <= 7.0):
            continue

        idxA = np.argmin(np.linalg.norm(points[:, :2] - pA, axis=1))
        idxB = np.argmin(np.linalg.norm(points[:, :2] - pB, axis=1))
        uA, vA = int(u_s[idxA]), int(v_s[idxA])
        uB, vB = int(u_s[idxB]), int(v_s[idxB])

        line_color = line_colors[idx % len(line_colors)]
        cv2.line(orig_img, (uA, vA), (uB, vB), line_color, 2)
        cv2.circle(orig_img, (uA, vA), 9, color_map_ordered[i], -1)
        cv2.circle(orig_img, (uB, vB), 9, color_map_ordered[j], -1)

        mid_u, mid_v = (uA + uB) // 2, (vA + vB) // 2
        text = f"{min_dist:.2f}m"
        cv2.putText(orig_img, text, (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(orig_img, text, (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 2, cv2.LINE_AA)

    # ------------------------------
    # 범례 추가 (항상 표시)
    # ------------------------------
    orig_img = add_legend(orig_img)

    # ------------------------------
    # 저장
    # ------------------------------
    save_path = os.path.join(output_dir, fname)
    cv2.imwrite(save_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))

    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"{fname} 처리 완료 - {elapsed:.2f}초")

# ------------------------------
# 전체 평균 처리 시간
# ------------------------------
if len(times) > 0:
    print(f"평균 처리 시간: {np.mean(times):.2f}초/장, 총 {len(times)}장 처리됨")
else:
    print("처리된 이미지가 없습니다.")
