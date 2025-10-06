# ================================================
# 14. NuScenes Mini Dataset 기반 일괄 처리 환경 세팅 (CAM_FRONT → CAM_RESULT)
# ================================================
#
# 목적:
#   DBSCAN 파라미터 튜닝 이전에, NuScenes Mini Dataset에 있는
#   CAM_FRONT 프레임을 순차적으로 처리하고
#   결과를 CAM_RESULT 폴더에 자동 저장하는 환경을 확립한다.
#
# 배경:
#   파라미터 실험을 안정적으로 진행하려면
#   전체 시퀀스에 동일한 설정이 적용되어야 한다.
#   따라서 본 단계에서는 전처리 일관성과 결과 비교 확인이 가능한
#   표준화된 “튜닝용 파이프라인”을 완성한다.
#
# 핵심 변경사항:
#   ① 입력 폴더: CAM_FRONT, 출력 폴더: CAM_RESULT
#   ② 좌우 범위(X): -5 ~ +5 (차량 중심부 기준 1차선 폭 + 여유 영역)
#   ③ 전체 프레임 자동 순회 및 결과 저장
#   ④ 평균 처리 시간 출력으로 효율성 검증
#
# 결과:
#   - CAM_RESULT 폴더 내에 프레임별 거리 분석 결과 이미지가 생성됨
#   - 이후 단계(파라미터 튜닝)는 본 환경을 기반으로 진행 예정
#
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/14_batch_tuning_environment_setup.py
# ================================================

import os
import time
import depth_pro
import torch
import numpy as np
import cv2
import itertools
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from sklearn.cluster import DBSCAN
import alphashape

# ------------------------------
# 0단계: 모델 초기화
# ------------------------------
model, transform = depth_pro.create_model_and_transforms()
model = model.to("cuda").half()
model.eval()

# 카메라 내외부 파라미터
K = np.array([
    [1266.417203046554,    0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0,    0.0,    1.0]
])
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# 입력/출력 경로
input_dir = "CAM_FRONT"
output_dir = "CAM_RESULT"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 1단계: CAM_FRONT 전체 프레임 순차 처리
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
    # 3단계: 포인트 클라우드 변환
    # ------------------------------
    vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u_flat, v_flat = us.flatten().astype(np.float32), vs.flatten().astype(np.float32)
    d_flat = depth.flatten().astype(np.float32)

    total = u_flat.size
    num_points = int(total * 0.1)
    idx = np.random.choice(total, num_points, replace=False)
    u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]

    # 전방 30m 이내, 하단 절반만 사용
    mask = (d_s <= 30) & (v_s >= h / 2)
    u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

    # 카메라 좌표계 변환
    x_n, y_n = (u_s - cx) / fx, (v_s - cy) / fy
    cam_right, cam_forward, cam_up = x_n * d_s, d_s, -y_n * d_s

    # 지면 필터 (-0.75~3m)
    z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
    cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]
    u_s, v_s = u_s[z_mask], v_s[z_mask]

    points = np.vstack([cam_right, cam_forward, cam_up]).T

    # ------------------------------
    # 4단계: DBSCAN + Alpha Shape
    # ------------------------------
    # 좌우 범위 좁힘 (-5~5m)
    mask = (
        (points[:, 0] >= -5) & (points[:, 0] <= 5) &
        (points[:, 1] >= 0) & (points[:, 1] <= 30)
    )
    points, u_s, v_s = points[mask], u_s[mask], v_s[mask]

    db = DBSCAN(eps=0.15, min_samples=25).fit(points[:, :2])
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
    # 5단계: 최소 거리 계산 및 시각화
    # ------------------------------
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    poly_keys = list(polygons.keys())

    for (i, j) in itertools.combinations(poly_keys, 2):
        polyA, polyB = polygons[i], polygons[j]
        min_dist = polyA.distance(polyB)
        pA, pB = nearest_points(polyA, polyB)
        pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])

        idxA = np.argmin(np.linalg.norm(points[:, :2] - pA, axis=1))
        idxB = np.argmin(np.linalg.norm(points[:, :2] - pB, axis=1))
        uA, vA = int(u_s[idxA]), int(v_s[idxA])
        uB, vB = int(u_s[idxB]), int(v_s[idxB])

        cv2.circle(orig_img, (uA, vA), 6, (0, 255, 0), -1)
        cv2.circle(orig_img, (uB, vB), 6, (0, 255, 0), -1)
        cv2.line(orig_img, (uA, vA), (uB, vB), (0, 255, 0), 2)

        mid_u, mid_v = (uA + uB) // 2, (vA + vB) // 2
        cv2.putText(orig_img, f"{min_dist:.2f} m", (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # ------------------------------
    # 6단계: 결과 저장 및 처리 시간 기록
    # ------------------------------
    save_path = os.path.join(output_dir, fname)
    cv2.imwrite(save_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))

    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"{fname} 처리 완료 - {elapsed:.2f}초")

# ------------------------------
# 7단계: 평균 처리 시간 출력
# ------------------------------
print(f"\n평균 처리 시간: {np.mean(times):.2f}초/장, 총 {len(times)}장 처리됨")
print(f"결과 저장 경로: {output_dir}")
