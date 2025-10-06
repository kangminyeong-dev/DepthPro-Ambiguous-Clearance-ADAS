# Depth Pro 탑뷰(Top-View) 변환 단계
# 목적: 카메라 좌표계 포인트 클라우드를 정제 후, 
#       전방(X-Y) 평면으로 투영하여 상단 시점(탑뷰)에서 분석 가능한 형태로 변환한다.
# 배경:
#   - 이전(06) 버전에서는 카메라 좌표계를 올바르게 정립했으나,
#     여전히 시각화가 3D 중심이라 실제 지면 패턴·전방 구조를 한눈에 보기 어려웠다.
#   - 이번 버전은 Z축 범위(-0.75~3m)로 정제한 후, 
#     X-Y 평면으로 투영하여 "도로 위 구조"를 탑뷰로 시각화한다.
# 핵심 개념:
#   1. 3D 공간에서 지면 기준 Z 범위를 필터링하여 노이즈 제거.
#   2. X(오른쪽)과 Y(전방) 축만 남겨 Top-View 투영.
#   3. 동일 포인트를 3D와 2D로 동시에 보여주어 시각적으로 대응 확인.
#   4. 색상은 전방 거리(cam_forward)에 따라 매핑하여 깊이감 유지.
# 결과:
#   - 왼쪽 그래프: Z축 포함된 3D 클라우드.
#   - 오른쪽 그래프: X-Y 평면 투영된 탑뷰.
#   - 전방(Forward) 방향의 깊이 정보가 색상으로 직관적으로 표현됨.
# 의미:
#   - Depth Pro의 3D 데이터를 실제 자율주행·지도 맵핑에서 활용 가능한 "탑뷰 기반" 데이터로 전환한 첫 버전.
#   - 이후 DBSCAN 클러스터링, Alpha-Shape, 거리 계산 등 응용 알고리즘의 입력으로 사용됨.
#
# 터미널 실행 명령어
# python kmy-depthpro-dev/07_topview_projection.py
#
# 2025-09-28 / Python 3.9.23 | FP16 GPU 환경 | Top-View Projection

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
    [1266.417203046554,    0.0, 816.2670197447984],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0,    0.0,    1.0]
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

# 카메라 좌표계
cam_right   = x_n * d_s        # X축: 오른쪽
cam_forward = d_s              # Y축: 전방
cam_up      = -y_n * d_s       # Z축: 위쪽 (부호 반전)

# ------------------------------
# 3단계: 지면 정제 (Z 범위 필터링)
# ------------------------------
z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]

# ------------------------------
# 4단계: 시각화 (3D + Top-View)
# ------------------------------
fig = plt.figure(figsize=(14, 6))

# 왼쪽: 3D 정제된 포인트
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(
    cam_right, cam_forward, cam_up,
    c=cam_forward, cmap="plasma", s=1, vmin=0, vmax=30
)
ax1.set_title("3D Point Cloud (Ground Refined)")
ax1.set_xlabel("Camera X (m, right)")
ax1.set_ylabel("Camera Y (m, forward)")
ax1.set_zlabel("Camera Z (m, up)")
ax1.set_xlim(-10, 10)
ax1.set_ylim(0, 15)
ax1.set_zlim(-0.75, 3)

# 오른쪽: 2D 탑뷰 투영
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(
    cam_right, cam_forward,
    c=cam_forward, cmap="plasma", s=1, vmin=0, vmax=30
)
ax2.set_title("Top-View Projection (X-Y plane)")
ax2.set_xlabel("Camera X (m, right)")
ax2.set_ylabel("Camera Y (m, forward)")
ax2.set_xlim(-10, 10)
ax2.set_ylim(0, 15)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()
