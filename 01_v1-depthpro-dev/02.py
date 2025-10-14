# python 01_v1-depthpro-dev/02.py

# ================================================
# Depth Pro GPU 추론 + 도로 정제 + 비도로 포인트 탑뷰 병렬 저장
# ================================================
import os
import time
import torch
import depth_pro
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# 입력/출력 폴더
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
    # 2단계: Depth Pro 추론 (GPU)
    # ------------------------------
    image, _, f_px = depth_pro.load_rgb(img_path)
    image_t = transform(image).to("cuda").half()
    with torch.no_grad():
        pred = model.infer(image_t, f_px=f_px)
    depth = pred["depth"].squeeze().detach().cpu().numpy()
    h, w = depth.shape

    # ------------------------------
    # 3단계: 포인트 클라우드 생성
    # ------------------------------
    vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    u_flat = us.flatten().astype(np.float32)
    v_flat = vs.flatten().astype(np.float32)
    d_flat = depth.flatten().astype(np.float32)

    num_points = int(u_flat.size * 0.05)
    idx = np.random.choice(u_flat.size, num_points, replace=False)

    u_s = u_flat[idx]
    v_s = v_flat[idx]
    d_s = d_flat[idx]

    # 전방 30m 이내 + 이미지 하단 절반
    mask = (d_s <= 30) & (v_s >= h / 2)
    u_s = u_s[mask]
    v_s = v_s[mask]
    d_s = d_s[mask]

    x_n = (u_s - cx) / fx
    y_n = (v_s - cy) / fy

    Xc = x_n * d_s
    Yc = y_n * d_s
    Zc = d_s
    points = np.stack([Xc, Yc, Zc], axis=1)

    # ------------------------------
    # 4단계: 지면 포인트 제거
    # ------------------------------
    grid_size = 0.2
    x_min, x_max = -10, 10
    z_min, z_max = 0, 30
    nx = int((x_max - x_min) / grid_size)
    nz = int((z_max - z_min) / grid_size)

    ix = np.clip(((points[:, 0] - x_min) / grid_size).astype(int), 0, nx - 1)
    iz = np.clip(((points[:, 2] - z_min) / grid_size).astype(int), 0, nz - 1)

    cell_dict = {}
    for i, (cx_i, cz_i) in enumerate(zip(ix, iz)):
        key = (cx_i, cz_i)
        if key not in cell_dict:
            cell_dict[key] = []
        cell_dict[key].append(i)

    density_th = 100
    height_std_th = 0.15
    min_points_cell = 10

    ground_mask = np.zeros(len(points), dtype=bool)
    for key, idxs in cell_dict.items():
        if len(idxs) < min_points_cell:
            ground_mask[idxs] = True
            continue
        local_y = points[idxs, 1]
        height_std = np.std(local_y)
        density = len(idxs)
        if (density < density_th) and (height_std < height_std_th):
            ground_mask[idxs] = True

    # 비도로 포인트만 남기기
    non_ground_pts = points[~ground_mask]
    if len(non_ground_pts) == 0:
        print(f"{fname}: 비도로 포인트 없음 → 원본 저장")
        orig_img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(output_dir, fname), orig_img)
        continue

    # ------------------------------
    # 5단계: 원본 + 탑뷰 병렬 시각화
    # ------------------------------
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # (왼쪽) 원본 이미지
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # (오른쪽) 비도로 포인트 탑뷰
    axes[1].scatter(non_ground_pts[:, 0], non_ground_pts[:, 2],
                    s=2, c="orange", alpha=0.6)
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(0, 30)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Camera X (m, right)")
    axes[1].set_ylabel("Depth Z (m, forward)")
    axes[1].set_title("Top-View (Non-ground Points)")

    plt.tight_layout()
    save_path = os.path.join(output_dir, fname.replace(".jpg", "_noground_topview.png"))
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    elapsed = time.time() - start_time
    times.append(elapsed)
    print(f"{fname} 처리 완료 - {elapsed:.2f}초")
    torch.cuda.empty_cache()

# ------------------------------
# 6단계: 평균 처리 시간 출력
# ------------------------------
if len(times) > 0:
    print(f"\n평균 처리 시간: {np.mean(times):.2f}초/장, 총 {len(times)}장 처리됨")
else:
    print("처리된 이미지가 없습니다.")
