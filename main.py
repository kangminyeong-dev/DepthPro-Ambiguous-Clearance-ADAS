import os
import time
import cv2
import numpy as np
import torch
import depth_pro
from sklearn.cluster import DBSCAN
import alphashape
from shapely.geometry import Polygon
from shapely.ops import nearest_points
import itertools
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

# ======================================================
# 경로 설정 (로컬용)
# ======================================================
img_paths  = "test_frames2"      # 입력 이미지 폴더
result_dir = "result_frames2-1"    # 결과 이미지 폴더
print("이미지 경로:", img_paths)

os.makedirs(img_paths, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ======================================================
# 디바이스 & FP16 설정 (DepthPro만)
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

###### SAM2 (Hydra config_dir 초기화) #####
GlobalHydra.instance().clear()

SAM2_CONFIG_DIR = "/home/kmy/adas_v2/DepthPro-Ambiguous-Clearance-ADAS/sam2/sam2/configs"
initialize_config_dir(config_dir=SAM2_CONFIG_DIR, version_base=None)

checkpoint = "/home/kmy/adas_v2/DepthPro-Ambiguous-Clearance-ADAS/sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg  = "sam2.1/sam2.1_hiera_s.yaml"

model_sam = build_sam2(model_cfg, checkpoint)
model_sam.to(device)
predictor = SAM2ImagePredictor(model_sam)

valid_exts = ('.jpg', '.jpeg', '.png')

###### ADAS 시작 ######
print("--------- ADAS 시작 ---------\n")
file_list = sorted(os.listdir(img_paths))

for img_file in file_list:
    if not img_file.lower().endswith(valid_exts):
        continue

    img_path = os.path.join(img_paths, img_file)
    file_name = os.path.basename(img_path)
    save_path = os.path.join(result_dir, file_name)

    print(f"Processing image: [{img_path}]\n")
    start_time = time.time()

    # -------------------------------------------------------
    # DepthPro 추론
    # -------------------------------------------------------
    img, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(img).to(device)
    if use_fp16:
        image_t = image_t.half()

    with torch.no_grad():
        prediction = model.infer(image_t, f_px=f_px)
        depth = prediction["depth"].squeeze().detach().cpu().numpy()
        h, w = depth.shape

    # -------------------------------------------------------
    # 원본 이미지 로드 (마지막 저장용)
    # -------------------------------------------------------
    orig_bgr = cv2.imread(img_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------
    # 포인트 클라우드 생성
    # -------------------------------------------------------
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

    mask_depth = (d_s <= 30)
    u_s, v_s, d_s = u_s[mask_depth], v_s[mask_depth], d_s[mask_depth]

    pts = np.stack([u_s, v_s], axis=1).astype(np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    u_ud = undistorted[:, 0, 0]
    v_ud = undistorted[:, 0, 1]

    x_n = (u_ud - cx) / fx
    y_n = (v_ud - cy) / fy
    Xc, Yc, Zc = x_n * d_s, -y_n * d_s, d_s
    points = np.stack([Xc, Yc, Zc], axis=1)

    # -------------------------------------------------------
    # 지면 제거(Grid)
    # -------------------------------------------------------
    grid_size = 0.2
    x_min, x_max, z_min, z_max = -10, 10, 0, 30
    nx, nz = int((x_max - x_min) / grid_size), int((z_max - z_min) / grid_size)

    ix = np.clip(((points[:, 0] - x_min) / grid_size).astype(int), 0, nx - 1)
    iz = np.clip(((points[:, 2] - z_min) / grid_size).astype(int), 0, nz - 1)

    cell_dict = {}
    for i, (cx_i, cz_i) in enumerate(zip(ix, iz)):
        key = (cx_i, cz_i)
        if key not in cell_dict:
            cell_dict[key] = []
        cell_dict[key].append(i)

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

    # -------------------------------------------------------
    # SAM2 기반 도로 정제
    # -------------------------------------------------------
    t0 = time.time()
    img_rgb = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    Z = points[:, 2]
    Y = points[:, 1]

    z_min_v, z_max_v = Z.min(), Z.max()
    z_bins = np.linspace(z_min_v, z_max_v, 5)
    auto_points = []

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

    # -------------------------------------------------------
    # SAM2 도로 + Grid 지면 통합 제거
    # -------------------------------------------------------
    u_idx = np.clip(u_s.astype(int), 0, w - 1)
    v_idx = np.clip(v_s.astype(int), 0, h - 1)
    is_road_sam = mask_clean[v_idx, u_idx] > 0

    final_mask = ground_mask | is_road_sam
    non_ground_pts = points[~final_mask]
    removed_pts = points[final_mask]
    print(f"[Ground+Road Filter] Total={len(points)} | Removed={np.sum(final_mask)} | Remain={len(non_ground_pts)}")

    # -------------------------------------------------------
    # DBSCAN + AlphaShape (X,Z 평면)
    # -------------------------------------------------------
    polygons = {}
    labels = None
    if len(non_ground_pts) > 0:
        db = DBSCAN(eps=0.4, min_samples=50).fit(non_ground_pts[:, [0, 2]])
        labels = db.labels_
        for label in set(labels):
            if label == -1:
                continue
            idx_lab = np.where(labels == label)[0]
            cluster_points = non_ground_pts[idx_lab][:, [0, 2]]
            if len(cluster_points) < 30:
                continue
            shape = alphashape.alphashape(cluster_points, alpha=0.5)
            if isinstance(shape, Polygon):
                polygons[label] = shape
    print(f"생성된 폴리곤 수: {len(polygons)}")

    # -------------------------------------------------------
    # 폴리곤 쌍 선택
    # -------------------------------------------------------
    from collections import OrderedDict
    final_pair = None
    if polygons:
        polygons = OrderedDict(sorted(polygons.items(), key=lambda x: int(x[0])))
        pairs = []
        filter_log = []

        z_axis = np.array([0, 1])

        def vector_angle(v1, v2):
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

        for i_p, j_p in itertools.combinations(polygons.keys(), 2):
            polyA, polyB = polygons[i_p], polygons[j_p]
            min_dist = polyA.distance(polyB)
            pA, pB = nearest_points(polyA, polyB)

            v = np.array([pB.x - pA.x, pB.y - pA.y])
            angle_to_z = vector_angle(v, z_axis)

            if not (20 <= angle_to_z <= 160):
                filter_log.append((int(i_p), int(j_p), min_dist, angle_to_z, "Too parallel to Z-axis"))
                continue

            pairs.append((i_p, j_p, min_dist, np.array(pA.coords[0]), np.array(pB.coords[0])))

        print("\n==== [Polygon Pair Filtering Report] ====")
        if len(filter_log) == 0:
            print("No pairs were filtered out.")
        else:
            for (a, b, dist_val, ang, reason) in filter_log:
                print(f"Filtered ({a},{b}) | Reason: {reason} | Dist={dist_val:.2f}m | angle={ang:.1f}°")
        print("=========================================\n")

        if len(pairs) == 0:
            print("No valid pairs remaining after filtering.")
        else:
            valid = pairs

            poly_depth = {}
            for l in polygons.keys():
                mask_l = (labels == l)
                if not np.any(mask_l):
                    continue
                pts_l = non_ground_pts[mask_l]
                z_min_l = np.min(pts_l[:, 2])
                x_mean = np.mean(pts_l[:, 0])
                center_bias = abs(x_mean)
                score_val = z_min_l + 0 * center_bias
                poly_depth[l] = score_val

            closest = min(poly_depth, key=poly_depth.get)
            print(f"[STAGE1] Closest polygon (Z+X-bias): {closest} (score={poly_depth[closest]:.3f})")

            candidates = [p for p in valid if closest in p[:2]]
            dropped = [p for p in valid if closest not in p[:2]]
            for (ii, jj, dist_val, pA_d, pB_d) in dropped:
                print(f"Dropped ({ii},{jj}) — Not containing front polygon (closest={closest})")

            def z_proximity_score(pair):
                i_pp, j_pp, _, pA_s, pB_s = pair
                if i_pp == closest:
                    x_obj, z_obj = pB_s
                else:
                    x_obj, z_obj = pA_s
                z_front = z_obj
                x_center = abs(x_obj)
                score_val = 0.5 * z_front + 0 * x_center
                return score_val

            final_pair = sorted(candidates, key=z_proximity_score)[0]
            i_sel, j_sel, min_dist_sel, pA_sel, pB_sel = final_pair
            print(f"[NEW SELECT] Closest pair: ({i_sel},{j_sel}) | min_dist={min_dist_sel:.2f} m")

    # -------------------------------------------------------
    # Final ADAS 시각화
    # -------------------------------------------------------
    if final_pair:
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = orig_img.shape

        # 1.9 m 기준으로 색상 결정
        color_map_ordered = {0: (50, 220, 50), 1: (255, 0, 0)}  # Green / Red
        car_width = 1.9  # 임계 너비 1.9 m

        def can_go_or_not(dist_val, width, cmap):
            return cmap[0] if dist_val >= width else cmap[1]

        i_fp, j_fp, min_dist_fp, pA_fp, pB_fp = final_pair
        base_color = can_go_or_not(min_dist_fp, car_width, color_map_ordered)

        def find_near_lowest_point(points_3d, ref_point_xz, radius=0.5, weight_dist=0.3):
            if len(points_3d) == 0:
                print("!!!points_3d None!!!")
                return np.array([ref_point_xz[0], 0, ref_point_xz[1]])

            dist_xz = np.linalg.norm(points_3d[:, [0, 2]] - ref_point_xz[None, :], axis=1)
            mask_local = dist_xz < radius
            candidate_points = points_3d[mask_local]
            candidate_dist = dist_xz[mask_local]

            if len(candidate_points) == 0:
                print("!!!candidate_points None!!! (radius fallback)")
                closest_idx = np.argmin(dist_xz)
                return points_3d[closest_idx]

            y_val = candidate_points[:, 1]
            dist_norm = candidate_dist / (np.max(candidate_dist) + 1e-6)
            score = -y_val - weight_dist * dist_norm

            best_idx = np.argmax(score)
            best_point = candidate_points[best_idx]

            print(f"find_near_lowest_point(radius={radius:.2f}): "
                  f"candidates={len(candidate_points)}, "
                  f"min_y={np.min(y_val):.3f}, chosen_y={best_point[1]:.3f}, "
                  f"min_dist={np.min(candidate_dist):.3f}, chosen_dist={candidate_dist[best_idx]:.3f}")

            return best_point

        pts_i = non_ground_pts[labels == i_fp]
        pts_j = non_ground_pts[labels == j_fp]

        if pA_fp[0] < pB_fp[0]:
            left_label, right_label = i_fp, j_fp
            ref_left, ref_right = pA_fp, pB_fp
        else:
            left_label, right_label = j_fp, i_fp
            ref_left, ref_right = pB_fp, pA_fp

        pts_left = non_ground_pts[labels == left_label]
        pts_right = non_ground_pts[labels == right_label]

        top_left_3d = find_near_lowest_point(pts_left, ref_left)
        top_right_3d = find_near_lowest_point(pts_right, ref_right)

        u_s_non_ground = u_s[~final_mask]
        v_s_non_ground = v_s[~final_mask]
        idxL = np.argmin(np.linalg.norm(non_ground_pts - top_left_3d, axis=1))
        idxR = np.argmin(np.linalg.norm(non_ground_pts - top_right_3d, axis=1))
        uL, vL = int(u_s_non_ground[idxL]), int(v_s_non_ground[idxL])
        uR, vR = int(u_s_non_ground[idxR]), int(v_s_non_ground[idxR])

        uA, vA = uL, vL
        uB, vB = uR, vR

        mid_u, mid_v = (uA + uB) // 2, (vA + vB) // 2
        text = f"{min_dist_fp:.2f} m"
        cv2.putText(orig_img, text, (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(orig_img, text, (mid_u, mid_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, base_color, 2, cv2.LINE_AA)

        if uA <= uB:
            uL, vL, uR, vR = uA, vA, uB, vB
        else:
            uL, vL, uR, vR = uB, vB, uA, vA
        uL_b, uR_b = int(w_img * 0.2), int(w_img * 0.8)
        vL_b = vR_b = h_img - 1

        roi_poly = np.array([[uL, vL], [uR, vR], [uR_b, vR_b], [uL_b, vL_b]], np.int32)
        cv2.polylines(orig_img, [roi_poly], True, base_color, 2)

        mask_roi = np.zeros((h_img, w_img))
        cv2.fillPoly(mask_roi, [roi_poly], 255)
        mask_roi = (mask_roi > 0)[..., None]

        AB = np.array([uB - uA, vB - vA], dtype=np.float32)
        a, b = AB[1], -AB[0]
        c_line = -(a * uA + b * vA)
        den = np.sqrt(a * a + b * b) + 1e-12
        max_d = abs(a * uL_b + b * vL_b + c_line) / den
        Y_grid, X_grid = np.mgrid[0:h_img, 0:w_img]
        dist_to_AB = np.abs(a * X_grid + b * Y_grid + c_line) / den
        t_ratio = np.clip(dist_to_AB / max_d, 0.0, 1.0)

        alpha_map = (0.08 + (0.5 - 0.08) * t_ratio)[..., None] * mask_roi
        blended = orig_img * (1 - alpha_map) + np.array(base_color, dtype=np.float32) * alpha_map
        orig_img = blended.astype(np.uint8)

        sorted_poly = roi_poly[np.argsort(roi_poly[:, 1])]
        bottom_mid = np.mean(sorted_poly[-2:], axis=0).astype(int)
        top_mid = np.mean(sorted_poly[:2], axis=0).astype(int)
        dash_len, gap_len = 15, 30
        line_vec = top_mid - bottom_mid
        line_len = int(np.linalg.norm(line_vec))
        direction = line_vec / (line_len + 1e-6)
        for d in range(0, line_len, dash_len + gap_len):
            start = (bottom_mid + direction * d).astype(int)
            end = (bottom_mid + direction * min(d + dash_len, line_len)).astype(int)
            cv2.line(orig_img, tuple(start), tuple(end), base_color, 2, cv2.LINE_AA)

        centroid_A = np.mean(np.array(polygons[i_fp].exterior.coords), axis=0)
        centroid_B = np.mean(np.array(polygons[j_fp].exterior.coords), axis=0)
        if centroid_A[0] < centroid_B[0]:
            centroid_left, centroid_right = centroid_A, centroid_B
            left_label2, right_label2 = i_fp, j_fp
        else:
            centroid_left, centroid_right = centroid_B, centroid_A
            left_label2, right_label2 = j_fp, i_fp

        mean_y_left = np.mean(non_ground_pts[labels == left_label2, 1])
        mean_y_right = np.mean(non_ground_pts[labels == right_label2, 1])

        centroid_left_3d = np.array([centroid_left[0], mean_y_left, centroid_left[1]])
        centroid_right_3d = np.array([centroid_right[0], mean_y_right, centroid_right[1]])

        idxL_c = np.argmin(np.linalg.norm(non_ground_pts - centroid_left_3d, axis=1))
        idxR_c = np.argmin(np.linalg.norm(non_ground_pts - centroid_right_3d, axis=1))
        uL_c, vL_c = int(u_s_non_ground[idxL_c]), int(v_s_non_ground[idxL_c])
        uR_c, vR_c = int(u_s_non_ground[idxR_c]), int(v_s_non_ground[idxR_c])

        centroid_left_3d_plus = np.array([centroid_left[0], 0.4, centroid_left[1]])
        centroid_right_3d_plus = np.array([centroid_right[0], 0.4, centroid_right[1]])

        idxL_plus = np.argmin(np.linalg.norm(non_ground_pts - centroid_left_3d_plus, axis=1))
        idxR_plus = np.argmin(np.linalg.norm(non_ground_pts - centroid_right_3d_plus, axis=1))
        uL_plus, vL_plus = int(u_s_non_ground[idxL_plus]), int(v_s_non_ground[idxL_plus])
        uR_plus, vR_plus = int(u_s_non_ground[idxR_plus]), int(v_s_non_ground[idxR_plus])

        sam_points = [
            ([uL_plus, vL_plus], [0, 0, 255], "Left-Top"),
            ([uL_c,    vL_c],    [0, 0, 255], "Left-Bottom"),
            ([uR_plus, vR_plus], [255, 0, 0], "Right-Top"),
            ([uR_c,    vR_c],    [255, 0, 0], "Right-Bottom")
        ]

        extra_points = []
        for lbl, centroid_3d, centroid_2d, color, label_name in [
            (left_label2,  centroid_left_3d,  centroid_left,  (0, 0, 255),   "Left"),
            (right_label2, centroid_right_3d, centroid_right, (255, 0, 0),   "Right")
        ]:
            poly = polygons[lbl]
            x_coords, z_coords = poly.exterior.xy
            x_min_p, x_max_p = np.min(x_coords), np.max(x_coords)
            z_min_p, z_max_p = np.min(z_coords), np.max(z_coords)

            corners = np.array([
                [x_min_p, z_min_p],
                [x_min_p, z_max_p],
                [x_max_p, z_min_p],
                [x_max_p, z_max_p]
            ])

            for cx_p, cz_p in corners:
                mid_x = (cx_p + centroid_2d[0]) / 2
                mid_z = (cz_p + centroid_2d[1]) / 2
                mid_3d = np.array([mid_x, centroid_3d[1], mid_z])
                idx_mid = np.argmin(np.linalg.norm(non_ground_pts - mid_3d, axis=1))
                u_mid, v_mid = int(u_s_non_ground[idx_mid]), int(v_s_non_ground[idx_mid])
                extra_points.append(([u_mid, v_mid], color, f"{label_name}-Mid"))

        sam_points = sam_points + extra_points

        img_np = orig_img.copy()
        predictor.set_image(img_np)
        overlay = img_np.copy()
        for (pt, color, label) in sam_points:
            masks_pt, scores_pt, _ = predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            best_idx_pt = np.argmax(scores_pt)
            mask_pt = masks_pt[best_idx_pt].astype(np.uint8)

            color_mask = np.zeros_like(img_np)
            color_mask[mask_pt > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.3, 0)

            # 점과 텍스트는 더 이상 그리지 않음 (객체 영역만 색으로 표현)
            # cv2.circle(overlay, tuple(pt), 8, color, -1, cv2.LINE_AA)
            # cv2.putText(overlay, label, (pt[0] + 10, pt[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        final_img = overlay
        ok = cv2.imwrite(save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    else:
        # 폴리곤 쌍 없으면 원본만 저장
        ok = cv2.imwrite(save_path, cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))

    print(f"이미지 한장 총 처리시간: {time.time()-start_time:.2f}초\n")
    print("-----------------------------------------------------------\n")

print("\n---------- 전체 종료 ----------\n")
