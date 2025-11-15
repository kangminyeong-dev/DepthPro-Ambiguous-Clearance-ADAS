# debug06.py
import time
import cv2
import torch
import depth_pro
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
import alphashape
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from collections import OrderedDict
import itertools

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

# ======================================================
# 입력 이미지 한 장
# ======================================================
img_path = "test_frames1/01.png"
print(f"[DEBUG] 입력 이미지: {img_path}")

# ======================================================
# 디바이스 & FP16 설정 (DepthPro용)
# ======================================================
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = (device.type == "cuda")
print(f"Device: {device}, FP16: {use_fp16}")

# ======================================================
# DepthPro 로드
# ======================================================
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device)
if use_fp16:
    model = model.half()
model.eval()

# ======================================================
# SAM2 초기화
# ======================================================
GlobalHydra.instance().clear()

SAM2_CONFIG_DIR = "/home/kmy/adas_v2/DepthPro-Ambiguous-Clearance-ADAS/sam2/sam2/configs"
initialize_config_dir(config_dir=SAM2_CONFIG_DIR, version_base=None)

checkpoint = "/home/kmy/adas_v2/DepthPro-Ambiguous-Clearance-ADAS/sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg  = "sam2.1/sam2.1_hiera_s.yaml"

model_sam = build_sam2(model_cfg, checkpoint)
model_sam.to(device)
predictor = SAM2ImagePredictor(model_sam)

# ======================================================
# DepthPro 추론
# ======================================================
img, _, f_px = depth_pro.load_rgb(img_path)
image_t = transform(img).to(device)
if use_fp16:
    image_t = image_t.half()

with torch.no_grad():
    prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"].squeeze().detach().cpu().numpy()
    h, w = depth.shape

print("[DEBUG] depth shape:", depth.shape)

# ======================================================
# 포인트 클라우드 생성
# ======================================================
K = np.array([
    [1.62963023e+03, 0.00000000e+00, 9.36652634e+02],
    [0.00000000e+00, 1.63835206e+03, 4.85700991e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

dist = np.array([0.19086768, -0.9629089, -0.00941831, -0.00365073, 1.81339113])

vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat, v_flat, d_flat = us.flatten(), vs.flatten(), depth.flatten()

num_points = int(u_flat.size * 0.05)
idx = np.random.choice(u_flat.size, num_points, replace=False)
u_s, v_s, d_s = u_flat[idx], v_flat[idx], d_flat[idx]

# 깊이 제한
mask = (d_s <= 30)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

# 왜곡 보정
pts = np.stack([u_s, v_s], axis=1).astype(np.float32).reshape(-1, 1, 2)
undistorted = cv2.undistortPoints(pts, K, dist, P=K)
u_ud = undistorted[:, 0, 0]
v_ud = undistorted[:, 0, 1]

# 카메라 좌표계로 변환
x_n = (u_ud - cx) / fx
y_n = (v_ud - cy) / fy
Xc, Yc, Zc = x_n * d_s, -y_n * d_s, d_s
points = np.stack([Xc, Yc, Zc], axis=1)

print("[DEBUG] 포인트 개수:", points.shape[0])

# ======================================================
# 지면 제거(Grid)
# ======================================================
grid_size = 0.2
x_min, x_max, z_min, z_max = -10, 10, 0, 30
nx, nz = int((x_max - x_min) / grid_size), int((z_max - z_min) / grid_size)

ix = np.clip(((points[:, 0] - x_min) / grid_size).astype(int), 0, nx - 1)
iz = np.clip(((points[:, 2] - z_min) / grid_size).astype(int), 0, nz - 1)

cell_dict = {}
for i_idx, (cx_i, cz_i) in enumerate(zip(ix, iz)):
    key = (cx_i, cz_i)
    if key not in cell_dict:
        cell_dict[key] = []
    cell_dict[key].append(i_idx)

density_th, height_std_th, min_points_cell = 100, 0.15, 10
ground_mask = np.zeros(len(points), dtype=bool)
for key, idxs in cell_dict.items():
    if len(idxs) < min_points_cell:
        ground_mask[idxs] = True
        continue
    local_y = points[idxs, 1]
    if np.std(local_y) < height_std_th:
        ground_mask[idxs] = True

print(f"Ground Filtering || Total: {len(points)}, Ground removed: {np.sum(ground_mask)}")

# ======================================================
# SAM2 기반 도로 정제 + 자동 포인트 선택
# ======================================================
t0 = time.time()
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
predictor.set_image(img_rgb)

Z = points[:, 2]
Y = points[:, 1]

z_min_v, z_max_v = Z.min(), Z.max()
z_bins = np.linspace(z_min_v, z_max_v, 5)  # 깊이 기준 4구간
auto_points = []

# 이미지 폭 기준 중앙 30% ~ 70% 영역
x_min_band = int(w * 0.3)
x_max_band = int(w * 0.7)

for k in range(4):
    z0, z1 = z_bins[k], z_bins[k + 1]
    mask_bin = (Z >= z0) & (Z < z1)
    mask_center = (u_s >= x_min_band) & (u_s <= x_max_band)
    mask_all = mask_bin & mask_center

    idx_bin = np.where(mask_all)[0]
    if len(idx_bin) == 0:
        continue

    # 해당 깊이 구간에서 Y가 가장 낮은 점(도로 후보)
    idx_min = idx_bin[np.argmin(Y[idx_bin])]
    u_pt = int(np.clip(u_s[idx_min], 0, w - 1))
    v_pt = int(np.clip(v_s[idx_min], 0, h - 1))
    auto_points.append([u_pt, v_pt])

if len(auto_points) == 0:
    auto_points = [[w // 2, int(h * 0.9)]]

input_point = np.array(auto_points)
input_label = np.ones(len(input_point))

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

best_idx = np.argmax(scores)
mask_base = masks[best_idx].astype(np.uint8)

kernel = np.ones((7, 7), np.uint8)
mask_clean = cv2.morphologyEx(mask_base, cv2.MORPH_OPEN, kernel)
t1 = time.time()
print(f"[SAM2 Road Segmentation] {t1 - t0:.3f} sec")
print("[DEBUG] auto_points:", auto_points)

# ======================================================
# SAM2 도로 + Grid 지면 통합 제거
# ======================================================
u_idx = np.clip(u_s.astype(int), 0, w - 1)
v_idx = np.clip(v_s.astype(int), 0, h - 1)
is_road_sam = mask_clean[v_idx, u_idx] > 0

final_mask     = ground_mask | is_road_sam
non_ground_pts = points[~final_mask]
removed_pts    = points[final_mask]

print(f"[Ground+Road Filter] Total={len(points)} | Removed={np.sum(final_mask)} | Remain={len(non_ground_pts)}")

# ======================================================
# DBSCAN + AlphaShape (X,Z 평면)
# ======================================================
polygons = {}
labels = None

if len(non_ground_pts) > 0:
    db = DBSCAN(eps=0.4, min_samples=50).fit(non_ground_pts[:, [0, 2]])
    labels = db.labels_
    for label in set(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        cluster_points = non_ground_pts[idx][:, [0, 2]]
        if len(cluster_points) < 30:
            continue
        shape = alphashape.alphashape(cluster_points, alpha=0.5)
        if isinstance(shape, Polygon):
            polygons[label] = shape

print(f"생성된 폴리곤 수: {len(polygons)}")

final_pair = None

if polygons:
    polygons = OrderedDict(sorted(polygons.items(), key=lambda x: int(x[0])))
    pairs = []
    filter_log = []

    # 기준축: Z축 (도로 진행 방향)
    z_axis = np.array([0, 1])  # X-Z 평면 기준 (0,1): 전방 방향

    def vector_angle(v1, v2):
        """두 벡터 사이 각도 계산 (degree)"""
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    for i_lbl, j_lbl in itertools.combinations(polygons.keys(), 2):
        polyA, polyB = polygons[i_lbl], polygons[j_lbl]
        min_dist = polyA.distance(polyB)
        pA, pB = nearest_points(polyA, polyB)

        # 최소거리 벡터 (X,Z 평면상)
        v = np.array([pB.x - pA.x, pB.y - pA.y])
        angle_to_z = vector_angle(v, z_axis)  # Z축과의 각도

        # 0° = Z축(앞뒤), 90° = X축(좌우) / 너무 Z축 평행이면 버림
        if not (20 <= angle_to_z <= 160):
            filter_log.append((int(i_lbl), int(j_lbl), min_dist, angle_to_z, "Too parallel to Z-axis"))
            continue

        pairs.append((i_lbl, j_lbl, min_dist, np.array(pA.coords[0]), np.array(pB.coords[0])))

    print("\n==== [Polygon Pair Filtering Report] ====")
    if len(filter_log) == 0:
        print("No pairs were filtered out.")
    else:
        for (a, b, dist_, ang, reason) in filter_log:
            print(f"Filtered ({a},{b}) | Reason: {reason} | Dist={dist_:.2f}m | angle={ang:.1f}°")
    print("=========================================\n")

    if len(pairs) == 0:
        print("No valid pairs remaining after filtering.")
    else:
        valid = pairs

        # [1차 정제] 가장 앞쪽(Z 작음) 폴리곤 선택
        poly_depth = {}
        for l in polygons.keys():
            mask_l = (labels == l)
            if not np.any(mask_l):
                continue
            pts_l = non_ground_pts[mask_l]
            z_min_l = np.min(pts_l[:, 2])
            x_mean = np.mean(pts_l[:, 0])
            center_bias = abs(x_mean)
            score = z_min_l + 0 * center_bias
            poly_depth[l] = score

        closest = min(poly_depth, key=poly_depth.get)
        print(f"[STAGE1] Closest polygon (Z+X-bias): {closest} (score={poly_depth[closest]:.3f})")

        # closest 포함된 쌍만 남기기
        candidates = [p for p in valid if closest in p[:2]]
        dropped = [p for p in valid if closest not in p[:2]]
        for (ii, jj, dist_, pA_d, pB_d) in dropped:
            print(f"Dropped ({ii},{jj}) — Not containing front polygon (closest={closest})")

        # [2차 정제] z_proximity_score
        def z_proximity_score(pair):
            i_p, j_p, _, pA_p, pB_p = pair
            if i_p == closest:
                x_obj, z_obj = pB_p
            else:
                x_obj, z_obj = pA_p

            z_front = z_obj
            x_center = abs(x_obj)
            score_ = 0.5 * z_front + 0 * x_center
            return score_

        final_pair = sorted(candidates, key=z_proximity_score)[0]
        i, j, min_dist, pA, pB = final_pair
        print(f"[NEW SELECT] Closest pair: ({i},{j}) | min_dist={min_dist:.2f} m")

# ======================================================
# Final ADAS 시각화 (Segmentation + ROI + 점선)
# ======================================================
if final_pair:
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape

    color_map_ordered = {0: (50, 220, 50), 1: (255, 0, 0)}  # Green: 진입 가능 / Red: 진입 불가
    car_width = 0.59

    def can_go_or_not(dist_val, width, cmap):
        return cmap[0] if dist_val >= width else cmap[1]

    i, j, min_dist, pA, pB = final_pair
    base_color = can_go_or_not(min_dist, car_width, color_map_ordered)

    # --------------------------------------------
    # 객체 상단점(ROI 윗점) 계산 함수
    # --------------------------------------------
    def find_near_lowest_point(points_3d, ref_point_xz, radius=0.5, weight_dist=0.3):
        """
        ref_point_xz 주변 반지름(radius) 내에서
        y가 낮고 (xz로도 가까운) 점을 선택.
        - weight_dist: xz 거리 영향 가중치 (0~1)
        """
        if len(points_3d) == 0:
            print("!!!points_3d None!!!")
            return np.array([ref_point_xz[0], 0, ref_point_xz[1]])

        # (1) XZ 거리 계산
        dist_xz = np.linalg.norm(points_3d[:, [0, 2]] - ref_point_xz[None, :], axis=1)

        # (2) radius 내 후보 필터링
        mask_local = dist_xz < radius
        candidate_points = points_3d[mask_local]
        candidate_dist = dist_xz[mask_local]

        # (3) 후보가 없으면 fallback: 가장 가까운 점
        if len(candidate_points) == 0:
            print("!!!candidate_points None!!! (radius fallback)")
            closest_idx = np.argmin(dist_xz)
            return points_3d[closest_idx]

        # (4) 후보점 스코어 계산 (낮은 y + 가까운 xz)
        y_val = candidate_points[:, 1]
        dist_norm = candidate_dist / (np.max(candidate_dist) + 1e-6)
        score_local = -y_val - weight_dist * dist_norm

        # (5) 최고 점수 선택
        best_idx = np.argmax(score_local)
        best_point = candidate_points[best_idx]

        # (6) 로그 출력
        print(
            f"find_near_lowest_point(radius={radius:.2f}): "
            f"candidates={len(candidate_points)}, "
            f"min_y={np.min(y_val):.3f}, chosen_y={best_point[1]:.3f}, "
            f"min_dist={np.min(candidate_dist):.3f}, chosen_dist={candidate_dist[best_idx]:.3f}"
        )

        return best_point

    # -------------------------------------------------------
    # ROI 상단 계산
    # -------------------------------------------------------
    # 두 폴리곤에 속한 포인트 추출
    pts_i = non_ground_pts[labels == i]
    pts_j = non_ground_pts[labels == j]

    # 좌우 구분 (x 기준)
    if pA[0] < pB[0]:
        left_label, right_label = i, j
        ref_left, ref_right = pA, pB
    else:
        left_label, right_label = j, i
        ref_left, ref_right = pB, pA

    pts_left = non_ground_pts[labels == left_label]
    pts_right = non_ground_pts[labels == right_label]

    top_left_3d = find_near_lowest_point(pts_left, ref_left)
    top_right_3d = find_near_lowest_point(pts_right, ref_right)

    # --------------------------------------------
    # 픽셀 좌표 변환
    # --------------------------------------------
    u_s_non_ground = u_s[~final_mask]
    v_s_non_ground = v_s[~final_mask]
    idxL = np.argmin(np.linalg.norm(non_ground_pts - top_left_3d, axis=1))
    idxR = np.argmin(np.linalg.norm(non_ground_pts - top_right_3d, axis=1))
    uL, vL = int(u_s_non_ground[idxL]), int(v_s_non_ground[idxL])  # ← 각 상단점 좌표
    uR, vR = int(u_s_non_ground[idxR]), int(v_s_non_ground[idxR])

    # --------------------------------------------
    # ROI 다각형 및 텍스트 시각화
    # --------------------------------------------
    uA, vA = uL, vL
    uB, vB = uR, vR

    mid_u, mid_v = (uA + uB) // 2, (vA + vB) // 2
    text = f"{min_dist:.2f} m"
    cv2.putText(
        orig_img,
        text,
        (mid_u, mid_v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        orig_img,
        text,
        (mid_u, mid_v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        base_color,
        2,
        cv2.LINE_AA,
    )

    # ROI 하단 좌표 (고정)
    if uA <= uB:
        uL, vL, uR, vR = uA, vA, uB, vB
    else:
        uL, vL, uR, vR = uB, vB, uA, vA
    uL_b, uR_b = w * 0.2, w * 0.8
    vL_b = vR_b = h - 1

    roi_poly = np.array([[uL, vL], [uR, vR], [uR_b, vR_b], [uL_b, vL_b]], np.int32)
    cv2.polylines(orig_img, [roi_poly], True, base_color, 2)

    # --- ROI 색상 그라데이션 채우기 ---
    mask_roi = np.zeros((h, w))
    cv2.fillPoly(mask_roi, [roi_poly], 255)
    mask_roi = (mask_roi > 0)[:, :, None]

    AB = np.array([uB - uA, vB - vA], dtype=np.float32)
    a, b = AB[1], -AB[0]
    c = -(a * uA + b * vA)
    den = np.sqrt(a * a + b * b) + 1e-12
    max_d = abs(a * uL_b + b * vL_b + c) / den
    Yg, Xg = np.mgrid[0:h, 0:w]
    dist_to_AB = np.abs(a * Xg + b * Yg + c) / den
    t_norm = np.clip(dist_to_AB / max_d, 0.0, 1.0)

    alpha_map = (0.08 + (0.5 - 0.08) * t_norm)[..., None] * mask_roi
    blended = orig_img * (1 - alpha_map) + np.array(base_color, dtype=np.float32) * alpha_map
    orig_img = blended.astype(np.uint8)

    # --- 중앙 점선 ---
    sorted_poly = roi_poly[np.argsort(roi_poly[:, 1])]
    bottom_mid = np.mean(sorted_poly[-2:], axis=0).astype(int)
    top_mid = np.mean(sorted_poly[:2], axis=0).astype(int)
    dash_len, gap_len = 15, 30
    line_vec = top_mid - bottom_mid
    line_len = int(np.linalg.norm(line_vec) + 1e-6)
    direction = line_vec / line_len
    for d_step in range(0, line_len, dash_len + gap_len):
        start = (bottom_mid + direction * d_step).astype(int)
        end = (bottom_mid + direction * min(d_step + dash_len, line_len)).astype(int)
        cv2.line(orig_img, tuple(start), tuple(end), base_color, 2, cv2.LINE_AA)

    # --- segmentation 포인트 ---
    centroid_A = np.mean(np.array(polygons[i].exterior.coords), axis=0)
    centroid_B = np.mean(np.array(polygons[j].exterior.coords), axis=0)
    if centroid_A[0] < centroid_B[0]:
        centroid_left, centroid_right = centroid_A, centroid_B
        left_label, right_label = i, j
    else:
        centroid_left, centroid_right = centroid_B, centroid_A
        left_label, right_label = j, i

    # 각 폴리곤의 실제 y(높이) 평균 계산
    mean_y_left = np.mean(non_ground_pts[labels == left_label, 1])
    mean_y_right = np.mean(non_ground_pts[labels == right_label, 1])

    centroid_left_3d = np.array([centroid_left[0], mean_y_left, centroid_left[1]])
    centroid_right_3d = np.array([centroid_right[0], mean_y_right, centroid_right[1]])

    idxL = np.argmin(np.linalg.norm(non_ground_pts - centroid_left_3d, axis=1))
    idxR = np.argmin(np.linalg.norm(non_ground_pts - centroid_right_3d, axis=1))
    uL, vL = int(u_s_non_ground[idxL]), int(v_s_non_ground[idxL])
    uR, vR = int(u_s_non_ground[idxR]), int(v_s_non_ground[idxR])

    centroid_left_3d_plus = np.array([centroid_left[0], 0.4, centroid_left[1]])
    centroid_right_3d_plus = np.array([centroid_right[0], 0.4, centroid_right[1]])

    idxL_plus = np.argmin(np.linalg.norm(non_ground_pts - centroid_left_3d_plus, axis=1))
    idxR_plus = np.argmin(np.linalg.norm(non_ground_pts - centroid_right_3d_plus, axis=1))
    uL_plus, vL_plus = int(u_s_non_ground[idxL_plus]), int(v_s_non_ground[idxL_plus])
    uR_plus, vR_plus = int(u_s_non_ground[idxR_plus]), int(v_s_non_ground[idxR_plus])

    sam_points = [
        ([uL_plus, vL_plus], [0, 0, 255], "Left-Top"),
        ([uL, vL], [0, 0, 255], "Left-Bottom"),
        ([uR_plus, vR_plus], [255, 0, 0], "Right-Top"),
        ([uR, vR], [255, 0, 0], "Right-Bottom"),
    ]

    extra_points = []
    for lbl, centroid_3d, centroid_2d, color, label_name in [
        (left_label, centroid_left_3d, centroid_left, (0, 0, 255), "Left"),
        (right_label, centroid_right_3d, centroid_right, (255, 0, 0), "Right"),
    ]:
        poly = polygons[lbl]
        x_coords, z_coords = poly.exterior.xy
        x_min_p, x_max_p = np.min(x_coords), np.max(x_coords)
        z_min_p, z_max_p = np.min(z_coords), np.max(z_coords)

        # 꼭짓점 4개 (x,z)
        corners = np.array(
            [
                [x_min_p, z_min_p],
                [x_min_p, z_max_p],
                [x_max_p, z_min_p],
                [x_max_p, z_max_p],
            ]
        )

        # 각 꼭짓점과 centroid(x,z) 중간점 계산
        for cx_p, cz_p in corners:
            mid_x = (cx_p + centroid_2d[0]) / 2
            mid_z = (cz_p + centroid_2d[1]) / 2

            # y는 centroid와 동일한 높이 사용
            mid_3d = np.array([mid_x, centroid_3d[1], mid_z])
            idx_mid = np.argmin(np.linalg.norm(non_ground_pts - mid_3d, axis=1))
            u_mid, v_mid = int(u_s_non_ground[idx_mid]), int(v_s_non_ground[idx_mid])

            extra_points.append(([u_mid, v_mid], color, f"{label_name}-Mid"))

    sam_points = sam_points + extra_points

    img_np = orig_img.copy()
    predictor.set_image(img_np)
    overlay = img_np.copy()
    for (pt, color, label_name) in sam_points:
        masks_sam, scores_sam, _ = predictor.predict(
            point_coords=np.array([pt]), point_labels=np.array([1]), multimask_output=True
        )
        best_idx_sam = np.argmax(scores_sam)
        mask_sam = masks_sam[best_idx_sam].astype(np.uint8)
        color_mask = np.zeros_like(img_np)
        color_mask[mask_sam > 0] = color
        overlay = cv2.addWeighted(overlay, 1, color_mask, 0.3, 0)
        cv2.circle(overlay, tuple(pt), 8, color, -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            label_name,
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    final_img = overlay

    plt.figure(figsize=(10, 4), dpi=150)
    plt.imshow(final_img)
    plt.axis("off")
    plt.title("Final ADAS")
    plt.show()
