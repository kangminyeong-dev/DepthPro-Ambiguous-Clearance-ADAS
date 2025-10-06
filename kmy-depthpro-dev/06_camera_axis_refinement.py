# Depth Pro 카메라 좌표계 정립 버전
# 목적: Depth Pro의 2D 깊이맵을 실제 카메라 기준 3D 공간으로 변환하고,
#       X(오른쪽), Y(전방), Z(위쪽) 방향을 물리적으로 올바르게 정렬한다.
# 배경:
#   - 이전(05) 버전은 시각적으로는 잘 보였으나,
#     내부 좌표계의 방향 의미가 실제 카메라 기준과 불일치했다.
#   - 이번 버전은 축 방향과 라벨을 물리 좌표계에 맞게 바로잡아
#     이후 수학적 처리(탑뷰 변환, 거리 계산 등)가 가능하도록 구조를 정리한다.
# 핵심 개념:
#   1. 카메라 좌표계 정의:
#        cam_right   → X축 (카메라 오른쪽)
#        cam_forward → Y축 (카메라 전방)
#        cam_up      → Z축 (카메라 위쪽, 부호 반전 적용)
#   2. 이미지 좌표계는 (왼→오, 위→아래)이므로, y축 방향을 반전해야 위쪽이 +가 된다.
#   3. 색상 매핑은 cam_forward(전방 거리)에 따라 설정하여,
#      가까운 물체는 보라색, 먼 물체는 노란색으로 표현한다.
#   4. 지면 중심(-0.75~3.0m) 구간만 남겨 불필요한 상부 포인트를 제거한다.
# 결과:
#   - 그림은 이전과 유사하지만, 좌표의 의미가 완전히 정립됨.
#   - 이후 탑뷰(Top-View) 변환, 경계 추정, 클러스터링 알고리즘을 적용할 수 있는 기반 완성.
# 의미:
#   - Depth Pro의 2D 예측 결과가 실제 3D 공간에서 해석 가능한 데이터로 전환된 첫 버전.
#
# 터미널 실행 명령어
# python kmy-depthpro-dev/06_camera_axis_refinement.py


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

# 정규화 좌표계 (카메라 내부 파라미터 보정)
x_n = (u_s - cx) / fx
y_n = (v_s - cy) / fy

# 카메라 좌표계 변환
cam_right   = x_n * d_s        # X축: 오른쪽
cam_forward = d_s              # Y축: 전방
cam_up      = -y_n * d_s       # Z축: 위쪽 (이미지 좌표 반전)

# ------------------------------
# 3단계: 시각화
# ------------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# Z축(cam_up) 범위 필터링 (-0.75 ~ 3)
z_mask = (cam_up >= -0.75) & (cam_up <= 3.0)
cam_right, cam_forward, cam_up = cam_right[z_mask], cam_forward[z_mask], cam_up[z_mask]

# 전방 거리 기준 색상 매핑
sc = ax.scatter(
    cam_right, cam_forward, cam_up,
    c=cam_forward, cmap="plasma", s=1, vmin=0, vmax=30
)

ax.set_xlabel("Camera X (m, right)")
ax.set_ylabel("Camera Y (m, forward)")
ax.set_zlabel("Camera Z (m, up)")

ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_zlim(-0.75, 3)

plt.colorbar(sc, label="Forward distance (0~30 m)")
plt.tight_layout()
plt.show()
