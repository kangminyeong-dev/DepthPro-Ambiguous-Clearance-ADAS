# Depth Pro 3D 시각화 개선 실험
# 목적: GPU 추론 이후 생성된 포인트 클라우드에서
#       지면 근처(Up-axis 기준 -0.75~3.0m)의 포인트만 남겨 시각적 구조를 정제한다.
# 배경:
#   - 이전 04 버전에서 FP16 메모리 최적화가 완료되어 여유가 생김.
#   - 이번에는 점 밀도를 10%로 높여 세밀한 구조를 관찰하면서,
#     실제 도로면에 해당하는 높이 구간만 필터링해 시각적 명확성을 확보.
# 주요 변화:
#   - 샘플링 비율: 1% → 10%
#   - Z축(Up-axis = -Yc) 필터링 추가 (-0.75~3.0)
#   - 지면 인접 포인트만 남겨 도로 중심 구조 관찰
# 의미:
#   - 단순 시각화에서 실제 공간 구조 분석 단계로 발전한 첫 코드
#   - 이후 탑뷰(Top-View) 변환과 클러스터링으로 확장될 기반 실험
#
# 터미널 실행 명령어
# python kmy-depthpro-dev/05_gpu_zaxis_filtering.py


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

total = u_flat.size
num_points = int(total * 0.1)
idx = np.random.choice(total, num_points, replace=False)

u_s = u_flat[idx]
v_s = v_flat[idx]
d_s = d_flat[idx]

mask = (d_s <= 30) & (v_s >= h / 2)
u_s, v_s, d_s = u_s[mask], v_s[mask], d_s[mask]

x_n = (u_s - cx) / fx
y_n = (v_s - cy) / fy

Xc = x_n * d_s
Yc = y_n * d_s
Zc = d_s

# ------------------------------
# 3단계: Z축(Up-axis) 범위 필터링 및 시각화
# ------------------------------
up_axis = -Yc
z_mask = (up_axis >= -0.75) & (up_axis <= 3.0)

Xc = Xc[z_mask]
Yc = Yc[z_mask]
Zc = Zc[z_mask]
up_axis = up_axis[z_mask]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(Xc, Zc, up_axis, c=Zc, cmap="plasma", s=1, vmin=0, vmax=30)

ax.set_xlabel("Camera X (m, right)")
ax.set_ylabel("Depth Y (m, forward)")
ax.set_zlabel("Camera Z (m, up)")

ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_zlim(-0.75, 3)

plt.colorbar(sc, label="Depth value (0~30 m)")
plt.tight_layout()
plt.show()
