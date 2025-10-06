# Depth Pro GPU 추론 및 시각화 코드
# 목적: 기존 CPU 추론 결과(.npz)를 불러오던 구조에서 벗어나,
#       모델을 직접 GPU(FP16)로 로드하고 실시간 추론한 깊이맵을 시각화한다.
# 배경:
#   - 처음엔 FP32(32비트 부동소수점)로 실행했으나,
#     Depth Pro 모델의 메모리 점유량이 매우 커서 GPU 메모리 부족(OOM)이 발생했다.
#   - 이를 계기로 FP16(half precision, 절반 정밀도)으로 전환해야 함을 처음으로 깨달았다.
#   - 이 버전부터 GPU 연산 최적화 및 실시간 추론 가능성에 대한 인식이 생겼다.
# 사용 환경: PyTorch CUDA (FP16), Depth Pro 공식 모델
# 주요 단계:
#   1. depth_pro.create_model_and_transforms() 로 모델 및 transform 로드
#   2. torch.no_grad() 환경에서 GPU 추론 수행
#   3. f_px(초점거리) 파라미터를 모델 내부 추론에 전달
#   4. 결과를 numpy로 변환 후 nuScenes 카메라 intrinsic 기반 역투영
#   5. 하단 절반 + 30m 이내 포인트만 시각화 (1% 샘플링)
# 의미:
#   - Python 내부에서 모델을 완전히 통합해 GPU 추론한 첫 버전
#   - FP16 최적화를 도입하면서 실제 디바이스(예: Jetson) 이식성의 기반이 마련됨
#   - 이후 단계에서 실시간화와 경량화 알고리즘의 출발점이 되는 버전

# 터미널 실행 명령어
# python kmy-depthpro-dev/03_gpu_inference_projection.py

from PIL import Image
import depth_pro
import torch
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1단계: GPU 추론
# ------------------------------
model, transform = depth_pro.create_model_and_transforms()

# 처음엔 FP32로 실행했으나 GPU 메모리 부족(OOM) 발생
# FP16으로 전환 후 정상 작동 확인됨
model = model.to("cuda").half()
model.eval()

image, _, f_px = depth_pro.load_rgb("data/test.jpg")
image = transform(image).to("cuda").half()

with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

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
num_points = int(total * 0.01)
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
# 3단계: 시각화
# ------------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(Xc, Zc, -Yc, c=Zc, cmap="plasma", s=1, vmin=0, vmax=30)

ax.set_xlabel("Camera X (m, right)")
ax.set_ylabel("Depth Z (m, forward)")
ax.set_zlabel("Camera Y (m, up)")

ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_zlim(-3, 3)

plt.colorbar(sc, label="Depth value (0~30 m)")
plt.tight_layout()
plt.show()
