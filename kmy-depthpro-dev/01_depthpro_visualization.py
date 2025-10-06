# Depth Pro의 첫 실험 코드
# 목적: CPU 추론으로 생성된 뎁스맵(.npz)을 직접 불러와 3D 점 구름 형태로 시각화한다.
# 사용 환경: Apple Depth Pro GitHub 공식 저장소를 git clone 후, 예제 명령어로 추론 결과를 얻은 상태.
# 이때 Depth Pro의 예시 이미지뿐 아니라 인터넷에 있는 실제 골목길 주행 전방 시점 이미지를 임의로 적용하여
# 실제 환경에서도 깊이 추정 결과가 어떻게 표현되는지 테스트했다.

# 주요 흐름:
#   1. 추론 결과 파일(.jpg, .npz) 경로 설정
#   2. 이미지 크기 확인 및 meshgrid 좌표 생성
#   3. 깊이맵(depth) 불러오기 및 좌표 변환
#   4. 일부 포인트 샘플링 (5%) 후 3D 시각화
#   5. 깊이값 필터링으로 노이즈 제거 후 컬러맵 표현

# 터미널 실행 명령어
# python kmy-depthpro-dev/01_depthpro_visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

result_dir = "result"
img_path = os.path.join(result_dir, "test1.jpg")
depth_path = os.path.join(result_dir, "test1.npz")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

data = np.load(depth_path)
depth = data["depth"]

ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

xs_flat = xs.flatten()
ys_flat = depth.flatten()
zs_flat = (h - ys).flatten()

total = len(xs_flat)
num_points = int(total * 0.05)
idx = np.random.choice(total, num_points, replace=False)

xs_sample = xs_flat[idx]
ys_sample = ys_flat[idx]
zs_sample = zs_flat[idx]

mask = (ys_sample <= 200) & (zs_sample <= h)
xs_sample = xs_sample[mask]
ys_sample = ys_sample[mask]
zs_sample = zs_sample[mask]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(xs_sample, ys_sample, zs_sample, c=ys_sample, cmap="plasma", s=1, vmin=0, vmax=30)

ax.set_xlim(0, w)
ax.set_ylim(4, 8)
ax.set_zlim(0, h)

plt.colorbar(sc, label="Depth value (0~50)")
plt.show()
