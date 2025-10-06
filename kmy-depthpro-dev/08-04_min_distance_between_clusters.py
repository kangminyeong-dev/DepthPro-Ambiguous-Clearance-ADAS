# ================================================
# 08-04 두 물체(클러스터) 간 최소 거리 계산 실험 (보완 설명 포함)
# ================================================
#
# 목적:
#   탑뷰 점군(top-view point cloud)에서 DBSCAN + Alpha Shape으로 구분된
#   두 물체 간의 최소 거리(minimum distance)를 수치적으로 계산하고,
#   그 결과를 시각화로 확인하기 위함이다.
#
# 배경:
#   앞선 단계(08-03)에서는 외곽 형태를 정의하는 데 집중했지만,
#   이번에는 물체 간 실제 간격을 계산하려는 시도로 확장되었다.
#
# 핵심 원리:
#   각 클러스터의 외곽선을 다각형(Polygon)으로 얻고,
#   각 다각형의 외곽선 좌표들(꼭짓점 vertex) 간의 유클리드 거리(norm)를 모두 비교하여
#   최소값을 찾는 방식으로 최소 거리를 계산한다.
#
# 한계점 (Limitations):
#   이 방식은 "다각형의 꼭짓점"만을 비교하므로,
#   실제로는 두 선분(segment) 사이의 최단 거리가 더 짧을 수 있음에도 불구하고
#   이를 반영하지 못한다.
#   즉, 다각형의 선분 중간부에서 가장 가까운 점이 존재하더라도
#   현재 구현에서는 그 점을 고려하지 않는다.
#   따라서 결과는 근사치(approximation)이며,
#   추후 shapely의 distance() 함수를 통한
#   Polygon-to-Polygon 최소거리 계산으로 보완할 수 있다.
#
# 터미널 실행 명령어:
# python kmy-depthpro-dev/08-04_min_distance_between_clusters.py
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
    (points[:,0] >= -10) & (points[:,0] <= 10) &
    (points[:,1] >= 0) & (points[:,1] <= 15)
)
points = points[mask]

# ------------------------------
# 3단계: DBSCAN 적용
# ------------------------------
db = DBSCAN(eps=0.08, min_samples=30).fit(points)
labels = db.labels_
unique_labels = set(labels)

# ------------------------------
# 4단계: 각 클러스터 외곽선(Alpha Shape) 저장
# ------------------------------
polygons = {}
for label in unique_labels:
    if label == -1:  # 노이즈
        continue
    
    cluster_points = points[labels == label]
    if len(cluster_points) < 100:
        continue

    alpha = 0.05
    shape = alphashape.alphashape(cluster_points, alpha)
    
    if isinstance(shape, Polygon):
        polygons[label] = shape

# ------------------------------
# 5단계: 두 클러스터 간 최소 거리 계산 (vertex 기반)
# ------------------------------
if len(polygons) >= 2:
    keys = list(polygons.keys())
    polyA = polygons[keys[0]]
    polyB = polygons[keys[1]]
    
    coordsA = np.array(polyA.exterior.coords)
    coordsB = np.array(polyB.exterior.coords)
    
    min_dist = float("inf")
    closest_pair = None

    # 다각형 꼭짓점(vertex) 기준 거리 계산
    for pA in coordsA:
        for pB in coordsB:
            dist = np.linalg.norm(pA - pB)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (pA, pB)
    
    print("두 물체(장애물) 사이 최소 거리 (vertex 기반):", f"{min_dist:.3f} m")
    print("※ 주의: 실제 최단점은 선분 중간일 수 있음 (근사치)")

    # ------------------------------
    # 6단계: 시각화
    # ------------------------------
    plt.figure(figsize=(8,8))

    # 각 클러스터 점 + 외곽선
    for label in polygons:
        cluster_points = points[labels == label]
        plt.scatter(cluster_points[:,0], cluster_points[:,1], s=2, alpha=0.5)
        x, y = polygons[label].exterior.xy
        plt.plot(x, y, 'r-', linewidth=1.5)

    # 최소 거리 선 시각화
    pA, pB = closest_pair
    plt.plot([pA[0], pB[0]], [pA[1], pB[1]], 'g--', linewidth=2)
    plt.scatter([pA[0], pB[0]], [pA[1], pB[1]], c='green', s=50, marker='x')

    plt.xlabel("Camera X (m, right)")
    plt.ylabel("Camera Y (m, forward)")
    plt.title(f"Min Distance ≈ {min_dist:.2f} m (vertex-based)")
    plt.xlim(-10, 10)
    plt.ylim(0, 15)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
