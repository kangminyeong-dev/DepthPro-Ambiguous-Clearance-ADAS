# debug02.py
import time
import cv2
import torch
import depth_pro
import numpy as np
import matplotlib.pyplot as plt

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
# 도로 마스크 + 시드 포인트 시각화
# ======================================================
overlay_road = img_rgb.copy()
overlay_road[mask_clean > 0] = [0, 255, 0]  # 도로 영역을 초록색으로
blended_road = cv2.addWeighted(img_rgb, 0.7, overlay_road, 0.3, 0)

vis = blended_road.copy()
for (u, v) in input_point:
    cv2.circle(vis, (int(u), int(v)), 6, (255, 0, 0), -1)  # 시드 포인트 (빨간 점)

plt.figure(figsize=(10, 5), dpi=500)
plt.imshow(vis)
plt.axis("off")
plt.title("SAM2 Road Segmentation Result")
plt.show()
