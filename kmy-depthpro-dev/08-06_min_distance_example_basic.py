# ================================================
# 08-06 최소 거리 연산 개념 예시 (직사각형 기반)
# ================================================
#
# 목적:
#   실제 탑뷰 데이터가 아닌 단순 사각형 2개를 사용해
#   shapely의 distance()와 nearest_points() 동작을
#   직관적으로 이해하기 위한 시각 실험이다.
#
# 개념 요약:
#   - 두 사각형(polyA, polyB)을 임의 각도로 회전시킨 뒤
#     꼭짓점 기준이 아닌 ‘선분 상의 점’을 포함한
#     실제 최소 거리 계산 과정을 시각적으로 확인한다.
#   - 이전 단계(08-04)의 “꼭짓점 거리 한계”를 보완함.
#
# 실행 환경:
#   Python 3.9.23 / shapely 2.0.4 / matplotlib 3.9.0
#   작성일: 2025-10-06
#
# 터미널 실행 명령어:
#   python kmy-depthpro-dev/08-06_min_distance_example_basic.py
# ================================================

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate
from shapely.ops import nearest_points

# ------------------------------
# 1단계: 두 개 직사각형 정의
# ------------------------------
polyA = Polygon([(0, 0), (0, 3), (2, 3), (2, 0)])     # 왼쪽 직사각형
polyB = Polygon([(4, 1), (4, 4), (6, 4), (6, 1)])     # 오른쪽 직사각형

# 중심 기준 회전 (A: 10도, B: 60도)
polyA = rotate(polyA, 10, origin='centroid')
polyB = rotate(polyB, 60, origin='centroid')

# ------------------------------
# 2단계: 최소 거리 및 최근접 점 계산
# ------------------------------
min_dist = polyA.distance(polyB)
p1, p2 = nearest_points(polyA, polyB)

print(f"두 다각형 사이 최소 거리: {min_dist:.3f} m")
print("최근접 점 A:", p1)
print("최근접 점 B:", p2)

# ------------------------------
# 3단계: 시각화 (plt.show)
# ------------------------------
plt.figure(figsize=(6, 6))

# 다각형 A (파란색)
x, y = polyA.exterior.xy
plt.plot(x, y, 'b-', linewidth=2)
plt.fill(x, y, 'blue', alpha=0.2, label="Polygon A")

# 다각형 B (빨간색)
x, y = polyB.exterior.xy
plt.plot(x, y, 'r-', linewidth=2)
plt.fill(x, y, 'red', alpha=0.2, label="Polygon B")

# 최소 거리 선분 및 점 표시
plt.plot([p1.x, p2.x], [p1.y, p2.y], 'g--', linewidth=2, label="Min Distance Line")
plt.scatter([p1.x, p2.x], [p1.y, p2.y], c='green', s=50, marker='x')

plt.title(f"Min Distance Example (Real = {min_dist:.2f} m)")
plt.xlabel("X axis (m)")
plt.ylabel("Y axis (m)")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()
