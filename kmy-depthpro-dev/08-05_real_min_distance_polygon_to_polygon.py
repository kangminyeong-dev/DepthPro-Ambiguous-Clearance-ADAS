# ================================================
# 08-05 다각형 간 실제 최소 거리 계산 (Polygon-to-Polygon)
# ================================================
#
# 목적:
#   Alpha Shape으로 정의된 두 물체(클러스터)의 외곽선을 기반으로
#   단순 꼭짓점(vertex) 간 근사 거리 대신,
#   실제 기하학적으로 가장 짧은 거리(Polygon-to-Polygon minimum distance)를 계산한다.
#
# 핵심 개선점:
#   이전 단계(08-04)에서는 각 다각형의 꼭짓점만 비교했지만,
#   이번 단계에서는 shapely의 distance()와 nearest_points() 함수를 사용해
#   다각형의 ‘선분 중간 지점(segment midpoint)’까지 포함한 **정확한 최소 거리**를 구했다.
#   즉, 선분 대 선분의 최단 거리를 산출하므로 이전보다 훨씬 정밀하다.
#
# 결과 해석:
#   출력되는 “Real_Min_Distance” 값은 두 다각형의 실제 공간적 최단거리(m)이며,
#   두 점(pA, pB)은 그 거리의 양 끝점을 나타낸다.
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/08-05_real_min_distance_polygon_to_polygon.py
# ================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import alphashape
from shapely.geometry import Polygon
from shapely.ops import nearest_points

# ------------------------------
# 1단계: 포인트 불러오기
# ------------------------------
points = np.loadtxt("topview_points.csv", delimiter=",", skiprows=1)

# ------------------------------
# 2단계: 관심 영역 정제 (탑뷰 범위)
# ------------------------------
mask = (
    (points[:,0] >= -30) & (points[:,0] <= 30) &
    (points[:,1] >= 0) & (points[:,1] <= 30)
)
points = points[mask]

# ------------------------------
# 3단계: DBSCAN 적용 (클러스터링)
# ------------------------------
db = DBSCAN(eps=0.08, min_samples=30).fit(points)
labels = db.labels_
unique_labels = set(labels)

# ------------------------------
# 4단계: Alpha Shape을 이용한 외곽 다각형 생성
# ------------------------------
polygons = {}
for label in unique_labels:
    if label == -1:  # 노이즈 제거
        continue
    
    cluster_points = points[labels == label]
    if len(cluster_points) < 100:
        continue
    
    alpha = 0.05
    shape = alphashape.alphashape(cluster_points, alpha)
    
    if isinstance(shape, Polygon):
        polygons[label] = shape

# ------------------------------
# 5단계: 다각형 간 최소 거리 계산 (Shapely 기반)
# ------------------------------
if len(polygons) >= 2:
    keys = list(polygons.keys())
    polyA = polygons[keys[0]]
    polyB = polygons[keys[1]]
    
    # Shapely의 polygon-to-polygon 거리 계산
    min_dist = polyA.distance(polyB)
    
    # 최근접 점 계산
    pA, pB = nearest_points(polyA, polyB)
    pA, pB = np.array(pA.coords[0]), np.array(pB.coords[0])
    
    print("두 장애물 사이 실제 최소 거리:", f"{min_dist:.3f} m")
    print(f"최근접 점 (Object A): {pA}")
    print(f"최근접 점 (Object B): {pB}")

    # ------------------------------
    # 6단계: 시각화
    # ------------------------------
    plt.figure(figsize=(8,8))
    
    for label in polygons:
        cluster_points = points[labels == label]
        plt.scatter(cluster_points[:,0], cluster_points[:,1], s=2, alpha=0.5)
        x, y = polygons[label].exterior.xy
        plt.plot(x, y, 'r-', linewidth=1.5)
    
    # 최소 거리 선분 표시
    plt.plot([pA[0], pB[0]], [pA[1], pB[1]], 'g--', linewidth=2)
    plt.scatter([pA[0], pB[0]], [pA[1], pB[1]], c='green', s=50, marker='x')
    
    plt.xlabel("Camera X (m, right)")
    plt.ylabel("Camera Y (m, forward)")
    plt.title(f"Real Min Distance = {min_dist:.2f} m (Polygon-to-Polygon)")
    plt.xlim(-10, 10)
    plt.ylim(0, 20)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
