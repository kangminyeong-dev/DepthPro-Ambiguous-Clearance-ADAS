# debug05.py
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

    for i, j in itertools.combinations(polygons.keys(), 2):
        polyA, polyB = polygons[i], polygons[j]
        min_dist = polyA.distance(polyB)
        pA, pB = nearest_points(polyA, polyB)

        # 최소거리 벡터 (X,Z 평면상)
        v = np.array([pB.x - pA.x, pB.y - pA.y])
        angle_to_z = vector_angle(v, z_axis)  # Z축과의 각도

        # 0° = Z축(앞뒤), 90° = X축(좌우) / 너무 Z축 평행이면 버림
        if not (20 <= angle_to_z <= 160):
            filter_log.append((int(i), int(j), min_dist, angle_to_z, "Too parallel to Z-axis"))
            continue

        pairs.append((i, j, min_dist, np.array(pA.coords[0]), np.array(pB.coords[0])))

    print("\n==== [Polygon Pair Filtering Report] ====")
    if len(filter_log) == 0:
        print("No pairs were filtered out.")
    else:
        for (a, b, dist, ang, reason) in filter_log:
            print(f"Filtered ({a},{b}) | Reason: {reason} | Dist={dist:.2f}m | angle={ang:.1f}°")
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
        for (ii, jj, dist, pA_d, pB_d) in dropped:
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
            score = 0.5 * z_front + 0 * x_center
            return score

        final_pair = sorted(candidates, key=z_proximity_score)[0]
        i, j, min_dist, pA, pB = final_pair
        print(f"[NEW SELECT] Closest pair: ({i},{j}) | min_dist={min_dist:.2f} m")

# -------------------------------------------------------
# 탑뷰 시각화 (전체 폴리곤 + 선택 쌍)
# -------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

# (좌) 전체 폴리곤
axs[0].set_title("Top-View: Non-Ground + Polygons (All)")
axs[0].set_xlabel("Camera X (right)")
axs[0].set_ylabel("Depth Z (forward)")
axs[0].scatter(non_ground_pts[:, 0], non_ground_pts[:, 2],
               s=3, c="gray", alpha=0.3, label="non-ground")

for lbl, poly in polygons.items():
    x, y = poly.exterior.xy
    axs[0].fill(x, y, alpha=0.3, label=f"Cluster {int(lbl)}")
    axs[0].plot(x, y, linewidth=2)
axs[0].axis("equal")
axs[0].legend(fontsize=7)

# (우) 최소 거리 쌍 확대 시각화
axs[1].set_title("Closest Polygon Pair (Zoomed)")
axs[1].set_xlabel("Camera X (right)")
axs[1].set_ylabel("Depth Z (forward)")
axs[1].scatter(non_ground_pts[:, 0], non_ground_pts[:, 2],
               s=2, c="lightgray", alpha=0.2)

if final_pair:
    polyA, polyB = polygons[i], polygons[j]
    for lbl, poly, color in [(i, polyA, "royalblue"),
                             (j, polyB, "darkorange")]:
        x, y = poly.exterior.xy
        axs[1].fill(x, y, alpha=0.4, color=color, label=f"Cluster {lbl}")
        axs[1].plot(x, y, color=color, linewidth=2)

    axs[1].plot([pA[0], pB[0]], [pA[1], pB[1]],
                "r--", linewidth=2, label=f"Min Dist: {min_dist:.2f} m")
    axs[1].scatter([pA[0], pB[0]], [pA[1], pB[1]],
                   c=["r", "r"], s=50, edgecolor="black", zorder=5)

    mid_x = (pA[0] + pB[0]) / 2
    mid_y = (pA[1] + pB[1]) / 2
    axs[1].text(mid_x, mid_y, f"{min_dist:.2f} m",
                color="red", fontsize=11, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(facecolor="white", alpha=0.6,
                          edgecolor="none", pad=1))

    pad = 2.0
    x_min = min(polyA.bounds[0], polyB.bounds[0]) - pad
    x_max = max(polyA.bounds[2], polyB.bounds[2]) + pad
    y_min = min(polyA.bounds[1], polyB.bounds[1]) - pad
    y_max = max(polyA.bounds[3], polyB.bounds[3]) + pad
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].axis("equal")
    axs[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
