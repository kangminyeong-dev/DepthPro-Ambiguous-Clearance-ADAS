# ================================================
# 08-02 순수 Delaunay 삼각분할 시각화 실험
# ================================================
#
# 목적:
#   Delaunay 삼각분할(Delaunay Triangulation)을 처음 적용해보며,
#   탑뷰 상의 점 데이터들이 어떤 방식으로 연결되는지
#   시각적으로 확인하기 위한 1차 실험이다.
#
# 배경:
#   당시에는 Delaunay 알고리즘을 직접 다뤄본 경험이 없었기 때문에,
#   일단 전체 포인트에 대해 삼각망(mesh)이 형성되는 과정을 보고자 했다.
#   하지만 모든 포인트가 자동으로 연결되어
#   물체의 외곽선과 관계없이 면이 꽉 차버렸고,
#   결과적으로 내가 의도한 “물체 테두리만의 연결 구조”는
#   잘 구분되지 않았다.
#
# 요약:
#   - 목적: Delaunay의 작동 원리 시각화
#   - 결과: 모든 점이 균일하게 삼각형으로 연결되어 구조가 복잡해짐
#   - 한계: 개별 물체나 영역 경계만 따로 추출하는 것은 어려움
#
# 터미널 실행 명령어:
# python kmy-depthpro-dev/08-02_delaunay_triangulation_test.py
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# ------------------------------
# 1단계: CSV 불러오기
# ------------------------------
csv_points = np.loadtxt("topview_points.csv", delimiter=",", skiprows=1)

# 좌표 분리
X = csv_points[:, 0]
Y = csv_points[:, 1]

# ------------------------------
# 2단계: Delaunay 삼각분할 수행
# ------------------------------
points = np.vstack((X, Y)).T  # (N, 2) 배열
tri = Delaunay(points)

# ------------------------------
# 3단계: 시각화
# ------------------------------
plt.figure(figsize=(8, 8))
plt.triplot(points[:, 0], points[:, 1], tri.simplices,
            color="gray", linewidth=0.3)
plt.scatter(points[:, 0], points[:, 1],
            c=points[:, 1], cmap="plasma", s=2)

plt.xlabel("Camera X (m, right)")
plt.ylabel("Camera Y (m, forward)")
plt.title("Delaunay Triangulation Visualization (Initial Test)")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(-10, 10)
plt.ylim(0, 15)
plt.tight_layout()
plt.show()
