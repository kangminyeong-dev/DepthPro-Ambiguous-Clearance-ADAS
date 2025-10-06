# ======================================================
# 20. 포인트 뿌리기 최적화 및 실시간 시각화 중심 버전
# ======================================================
#
# 목적:
#   ① 포인트 정제 순서를 “필터 후 샘플링”으로 바꿔서 연산 효율 개선
#   ② 결과를 파일로 저장하지 않고 실시간 시각화로 경량화
#   ③ 향후 지면 높이를 히스토그램 기반으로 자동 보정하는 방향으로 확장 가능성 확보
#
# 추가 설명:
#   - 기존엔 전체 포인트를 뿌리고 나서 정제했지만,
#     이제는 먼저 정제(mask)한 뒤 필요한 부분만 샘플링함.
#   - 이는 DBSCAN의 입력 포인트 품질을 높여 연산량을 줄이는 효과가 있음.
#   - 파라미터 튜닝도 최적화 계속 진행중(다른 환경에서도 잘 될지 확인 필요)
#   - 장착 환경이 바뀌어도 자동으로 지면 기준을 재설정하는 기능은
#     향후 cam_up 분포 히스토그램 분석으로 구현 예정.
#     예: 지면 구간을 하위 90%~0.0 범위로 설정.
#   - 보드 환경용 모델 가공 필요 ONNX
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/20_point_optimize_realtime.py
# ======================================================

import os
import time
import depth_pro
import torch
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

# 카메라 내외부 파라미터
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# 입력 경로
input_dir = "CAM_FRONT"

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
    # 추론 (Depth Pro)
    # ------------------------------
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(image).to("cuda").half()
    with torch.no_grad():
        prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"].squeeze().detach().cpu().numpy()
    h, w = depth.shape

    # ------------------------------
    # 포인트클라우드 변환 (정제 우선)
# ------------------------------
    vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u_flat, v_flat = us.flatten().astype(np.float32), vs.flatten().astype(np.float32)
    d_flat = depth.flatten().astype(np.float32)

    # (1) 우선 필터링: 거리·시야·하단부 제한
    mask = (d_flat <= 25) & (v_flat >= h / 2)
    u_r, v_r, d_r = u_flat[mask], v_flat[mask], d_flat[mask]

    # (2) 샘플링
    total = u_r.size
    num_points = int(total * 0.20)
    if total > num_points:
        idx = np.random.choice(total, num_points, replace=False)
        u_s, v_s, d_s = u_r[idx], v_r[idx], d_r[idx]
    else:
        u_s, v_s, d_s = u_r, v_r, d_r

    # (3) 카메라 좌표계 변환
    x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
    cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

    # (4) 지면 필터링 (임시 고정, 향후 자동 보정 예정)
    z_mask = (cam_up >= -0.5) & (cam_up <= 0.0)
    cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
    u_s, v_s, d_s = u_s[z_mask], v_s[z_mask], d_s[z_mask]

    # (5) XY 영역 제한
    xy_mask = (
        (cam_right >= -4.5) & (cam_right <= 4.5) &
        (cam_forward >= 0) & (cam_forward <= 25)
    )
    cam_right, cam_forward, cam_up = cam_right[xy_mask], cam_forward[xy_mask], cam_up[xy_mask]
    u_s, v_s, d_s = u_s[xy_mask], v_s[xy_mask], d_s[xy_mask]

    points = np.vstack([cam_right, cam_forward, cam_up]).T

    # ------------------------------
    # 범례 함수
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
    # DBSCAN + Alpha Shape
    # ------------------------------
    polygons = {}
    if len(points) > 0:
        db = DBSCAN(eps=0.5, min_samples=300).fit(points[:, :2])
        labels = db.labels_
        for label in set(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label, :2]
            if len(cluster_points) < 100:
                continue
            shape = alphashape.alphashape(cluster_points, alpha=0.01)
            if isinstance(shape, Polygon):
                polygons[label] = shape

    # ------------------------------
    # 가까운 3개 다각형 선택
    # ------------------------------
    if len(polygons) > 0:
        poly_with_dist = []
        for label, poly in polygons.items():
            centroid = np.mean(np.array(poly.exterior.coords), axis=0)
            dist = np.linalg.norm(centroid)
            poly_with_dist.append((label, poly, dist))
        poly_with_dist.sort(key=lambda x: x[2])
        poly_with_dist = poly_with_dist[:3]
        polygons = {idx: poly for idx, (label, poly, _) in enumerate(poly_with_dist)}

    # ------------------------------
    # 색상·라인 정의
    # ------------------------------
    color_map_ordered = {0: (255, 255, 0), 1: (0, 0, 255), 2: (139, 69, 19)}
    line_colors = [(124, 252, 0), (144, 238, 144), (0, 255, 0)]

    # ------------------------------
    # 최소 거리 계산 및 실시간 시각화
    # ------------------------------
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = add_legend(orig_img)

    if len(polygons) > 0:
        poly_keys = list(polygons.keys())
        for idx, (i, j) in enumerate(itertools.combinations(poly_keys, 2)):
            polyA, polyB = polygons[i], polygons[j]
            min_dist = polyA.distance(polyB)
            pA, pB = nearest_points(polyA, polyB)
            pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])
            line = LineString([pA, pB])

            # 중간 객체 교차 제거
            skip = any(line.intersects(polyC) for k, polyC in polygons.items() if k not in (i, j))
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
    # 실시간 시각화 출력
    # ------------------------------
    cv2.imshow("Result", cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)
    if key == 27:  # ESC 키 종료
        break

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

cv2.destroyAllWindows()
