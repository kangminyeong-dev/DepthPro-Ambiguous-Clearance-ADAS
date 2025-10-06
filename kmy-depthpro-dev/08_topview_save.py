# Depth Pro 탑뷰(Top-View) 데이터 저장 버전
#
# 목적:
#   이 시점에서는 이미 전처리된 탑뷰 투영이 성공적으로 구현된 상태였다.
#   이후의 핵심 과제는 "탑뷰 상의 두 물체(또는 클러스터) 사이 최소 거리"를
#   어떻게 계산할 것인가를 구체화하는 것이었기 때문에,
#   일단 데이터를 시각화하고 외부 파일로 안전하게 저장하여
#   후속 알고리즘 실험에 활용할 수 있도록 한 것이다.
#
# 배경:
#   전 단계까지는 Depth Pro를 통한 거리 추론과 포인트 클라우드 시각화에 초점이 있었다.
#   하지만 본격적인 거리 기반 인식(예: 장애물 간 간격, 주행 가능 폭 계산)을 시도하려면
#   데이터셋 형태로 관리 가능한 탑뷰 점군이 필요했다.
#   따라서 본 코드는 '탑뷰 결과를 분석 가능한 상태로 내보낸 첫 버전'이다.
#
# 핵심 기능:
#   1. FP16 기반 GPU 추론 → 메모리 효율 확인 (~3.8GB 수준)
#   2. 포인트 클라우드 변환 및 Z축(-0.75~3.0m) 범위 정제
#   3. 탑뷰(X-Y 평면) 투영 및 시각화
#   4. 시각화 결과를 PNG로 저장
#   5. 포인트 좌표를 CSV/NPY로 저장 (후속 거리 계산, 클러스터링 실험용)
#
# 결과물:
#   - topview_projection.png : 탑뷰 시각화 이미지
#   - topview_points.csv      : (X, Y) 평면 좌표 데이터
#   - topview_points.npy      : 동일 데이터를 Numpy 배열로 저장
#
# 의의:
#   이 시점은 단순 시각화에서 “데이터 기반 거리 분석”으로 넘어가기 전환점이었다.
#   이후 단계부터 이 데이터로부터 두 물체 간 최소거리 계산을 어떻게 모델링할지를 집중적으로 고민하게 된다.
#
# 터미널 실행 명령어:
# python kmy-depthpro-dev/08_topview_save.py
#

from PIL import Image
import depth_pro
import torch
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1단계: GPU 추론 + 메모리 추적
# ------------------------------
model, transform = depth_pro.create_model_and_transforms()
model = model.to("cuda").half()
model.eval()

image, _, f_px = depth_pro.load_rgb("data/test.jpg")
image = transform(image).to("cuda").half()

torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

max_mem = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Max GPU memory used: {max_mem:.2f} MB")

depth = prediction["depth"]
depth_np = depth.squeeze().detach().cpu().numpy()
h, w = depth_np.shape

# ------------------------------
# 2단계: 포인트 클라우드 변환
# ------------------------------
K = np.array([
    [1266.417203046554, 0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0]
])

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
u_flat = us.flatten().astype(np.float32)
v_flat = vs.flatten().astype(np.float32)
d_flat = depth_np.flatten().astype(np.float32)

# 랜덤 샘플링 (10%)
total = u_flat.size
num_points = int(total * 0.1)
idx = np.random.choice(total, num_points, replace=False)

u_s = u_flat[idx]
v_s = v_flat[idx]
d_s = d_flat[idx]

# 전방(30m 이내) + 이미지 하단부만 사용
mask = (d_s <= 30) & (v_s >= h / 2)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

# 정규화 좌표계
x_n = (u_s - cx) / fx
y_n = (v_s - cy) / fy

# 카메라 좌표계 변환
cam_right   = x_n * d_s        # 좌우 (X축)
cam_forward = d_s              # 전방 (Y축)
cam_up      = -y_n * d_s       # 위쪽 (Z축, 부호 반전)

# ------------------------------
# 3단계: 지면 정제 (Z 범위 필터링)
# ------------------------------
z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]

# ------------------------------
# 4단계: 시각화 및 저장
# ------------------------------
fig = plt.figure(figsize=(14, 6))

# 왼쪽: 3D 포인트
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(cam_right, cam_forward, cam_up, c=cam_forward, cmap="plasma", s=1, vmin=0, vmax=30)
ax1.set_title("3D Point Cloud (Ground Refined)")
ax1.set_xlabel("Camera X (m, right)")
ax1.set_ylabel("Camera Y (m, forward)")
ax1.set_zlabel("Camera Z (m, up)")
ax1.set_xlim(-10, 10)
ax1.set_ylim(0, 15)
ax1.set_zlim(-0.75, 3)

# 오른쪽: 탑뷰 투영
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(cam_right, cam_forward, c=cam_forward, cmap="plasma", s=1, vmin=0, vmax=30)
ax2.set_title("Top-View Projection (X-Y plane)")
ax2.set_xlabel("Camera X (m, right)")
ax2.set_ylabel("Camera Y (m, forward)")
ax2.set_xlim(-10, 10)
ax2.set_ylim(0, 15)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("topview_projection.png", dpi=300)
plt.show()

# ------------------------------
# 5단계: 데이터 저장 (CSV + NPY)
# ------------------------------
points = np.vstack([cam_right, cam_forward]).T  # (N, 2) 배열

np.savetxt("topview_points.csv", points, delimiter=",", header="X,Y", comments="")
np.save("topview_points.npy", points)

print("탑뷰 이미지: topview_projection.png")
print("탑뷰 좌표 CSV: topview_points.csv")
print("탑뷰 좌표 NPY: topview_points.npy")
