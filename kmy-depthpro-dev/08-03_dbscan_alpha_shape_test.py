# ================================================
# 08-03 DBSCAN + Alpha Shape 기반 객체 외곽 추출 실험
# ================================================
#
# 목적:
#   탑뷰 점군(top-view point cloud)에서 실제 ‘물체’처럼 보이는 영역을
#   군집화(Clustering)하여 구분하고,
#   각 물체의 점들을 Alpha Shape으로 연결해
#   형태(Shape) 단위로 외곽선을 시각화하는 실험이다.
#
# 배경:
#   이전 단계에서는 모든 점이 무작위로 연결되어
#   물체의 경계를 식별하기 어려웠다.
#   따라서 이번에는 먼저 DBSCAN으로 “밀도가 높은 영역”만 분리해
#   각 영역을 물체처럼 정의하고,
#   그 내부 점들을 Alpha Shape으로 연결해 외곽 형태를 추출했다.
#
# Alpha Shape 개념 요약:
#   - Alpha Shape은 Delaunay Triangulation(들로네 삼각분할)로부터 파생된 알고리즘 라이브러리이다.
#   - 모든 점을 연결하는 Delaunay 삼각망을 먼저 만든 뒤,
#     각 삼각형의 외접원 반지름을 계산해,
#     α(알파) 값보다 큰 삼각형을 제거함으로써
#     실제 물체 윤곽에 해당하는 삼각형만 남긴다.
#   - 즉, Alpha Shape은 Delaunay 결과에서 “불필요한 연결을 거르고
#     외곽 형태만 남긴 정제된 버전”이라고 할 수 있다.
#
# 구현 요약:
#   1. CSV 파일에서 탑뷰 점 데이터 불러오기
#   2. 관심 구역(X: ±10m, Y: 0~15m) 필터링
#   3. DBSCAN(eps=0.08, min_samples=30)으로 군집화
#   4. 각 클러스터 내 점들을 Alpha Shape으로 연결
#   5. 각 물체의 윤곽선을 시각화
#
# 실험 결과:
#   - 밀집 영역이 자연스럽게 분리되어 각각 하나의 물체처럼 표현됨
#   - Alpha Shape으로 외곽선이 매끄럽게 이어지며 형태적 일관성이 확보됨
#   - eps(군집 민감도)와 alpha(형태 민감도)가
#     결과에 큰 영향을 미친다는 점을 체감한 첫 실험이었다.
#
# 터미널 실행 명령어:
# python kmy-depthpro-dev/08-03_dbscan_alpha_shape_test.py
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import alphashape
from shapely.geometry import Polygon

# ------------------------------
# 1단계: 포인트 불러오기
# ------------------------------
points = np.loadtxt("topview_points.csv", delimiter=",", skiprows=1)

# ------------------------------
# 2단계: 관심 영역 정제 (탑뷰 범위)
# ------------------------------
mask = (
    (points[:, 0] >= -10) & (points[:, 0] <= 10) &   # X 범위
    (points[:, 1] >= 0) & (points[:, 1] <= 15)       # Y 범위
)
points = points[mask]

# ------------------------------
# 3단계: DBSCAN 적용 (엄격 설정)
# ------------------------------
db = DBSCAN(eps=0.08, min_samples=30).fit(points)
labels = db.labels_
unique_labels = set(labels)

# ------------------------------
# 4단계: Alpha Shape 외곽 추출 및 시각화
# ------------------------------
plt.figure(figsize=(8, 8))

for label in unique_labels:
    if label == -1:  # 노이즈는 제외
        continue

    cluster_points = points[labels == label]

    # 작은 클러스터는 제거
    if len(cluster_points) < 100:
        continue

    # Alpha Shape으로 윤곽선 생성
    alpha = 0.05
    shape = alphashape.alphashape(cluster_points, alpha)

    # 클러스터 점 시각화
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=1)

    # 외곽선 시각화
    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        plt.plot(x, y, 'r-', linewidth=1.5)

plt.xlabel("Camera X (m, right)")
plt.ylabel("Camera Y (m, forward)")
plt.title("Top-View Cluster Shapes (DBSCAN + Alpha Shape)")
plt.xlim(-10, 10)
plt.ylim(0, 15)
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()
