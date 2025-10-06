# Depth Pro GPU 메모리 사용량 추적 코드
# 목적: FP16 전환 이후 Depth Pro 모델이 실제 추론 시 사용하는 GPU 메모리량을
#       torch.cuda 모듈을 통해 정량적으로 측정하고 비교 관점에서 해석한다.
# 배경:
#   - 이전 03 버전에서 FP32 실행 시 OOM(Out of Memory) 오류를 경험했다.
#   - FP16으로 전환 후 정상 작동함을 확인했으며, 그 효과를 수치로 확인하기 위해
#     이번 버전에서 GPU 메모리 사용량을 직접 측정했다.
# 결과:
#   - 터미널 출력: Max GPU memory used: 3796.27 MB
#   - 즉, FP16(half precision) 기준 약 3.8 GB의 메모리를 사용.
#   - 일반적으로 FP32는 FP16의 2배 용량을 차지하므로,
#     원래 FP32 기준으로는 약 7.5 GB 수준일 것으로 추정된다.
#   - 이를 통해 Jetson Orin 8 GB 같은 실제 환경에서는 무거운 모델 기준 FP16으로 안정적인 추론이 가능하다는 결론에 도달했다.
#   - 또한 INT8(8비트 정수형 양자화) 방식의 존재도 확인했으나,
#     실제 딥러닝 모델 추론에서는 FP16이 호환성과 안정성 면에서 더 널리 통용된다는 점도 학습했다.
# 추가 기능:
#   - torch.cuda.reset_peak_memory_stats(): GPU 메모리 통계 초기화
#   - torch.cuda.max_memory_allocated(): 최대 사용량(MB 단위) 기록
#   - print()로 실시간 결과 출력
# 구조:
#   - 추론 및 시각화 흐름은 03 버전과 동일
#   - 일부 라벨이 임시로 변경되었으나 기능에는 영향 없음
# 의미:
#   - FP16 최적화 효과를 수치로 입증한 첫 실험 버전
#   - FP16 vs FP32 vs INT8 간의 구조적 차이를 체계적으로 이해하게 된 계기
#   - 이후 Jetson 이식과 실시간 최적화 전략 설계의 기초가 된 단계
#
# 터미널 실행 명령어
# python kmy-depthpro-dev/04_gpu_memory_tracking.py


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

max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
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
ax.set_ylabel("Depth Y (m, forward)")  # 임시 라벨
ax.set_zlabel("Camera Z (m, up)")      # 임시 라벨

ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_zlim(-3, 3)

plt.colorbar(sc, label="Depth value (0~30 m)")
plt.tight_layout()
plt.show()
