# ================================================
# 08-01 기존 탑뷰 데이터 로드 및 시각화
# ================================================
#
# 목적:
#   이전 단계(08_topview_save.py)에서 저장한 탑뷰 좌표 파일(CSV, NPY)을
#   불러와서 저장 포맷의 일관성을 검증하고,
#   시각적으로 다시 투영했을 때 원래 탑뷰 결과와 동일한지 확인한다.
#
# 배경:
#   전 단계에서 "탑뷰 투영 결과를 분석 가능한 형태로 저장"했지만,
#   데이터가 정상적으로 저장·로드되는지, 좌표계가 깨지지 않았는지
#   반드시 검증할 필요가 있었다.
#
# 핵심 기능:
#   1. CSV 파일 로드 및 shape/예시 출력
#   2. NPY 파일 로드 및 shape/예시 출력
#   3. 두 결과 비교 (좌표 일치 여부 확인)
#   4. 시각화 (탑뷰 재현)
#
# 결론:
#   - CSV와 NPY 모두 동일한 좌표 세트를 포함하며,
#   - 컬러맵과 위치가 이전 탑뷰 이미지와 동일하다면 저장 과정이 완전하게 수행된 것.
#
# 터미널 실행 명령어:
# python kmy-depthpro-dev/08-01_load_and_visualize_topview.py
#
# 2025-09-30 / Python 3.9.23 / FP16 환경 / Top-View Verification

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1단계: CSV 불러오기
# ------------------------------
csv_points = np.loadtxt("topview_points.csv", delimiter=",", skiprows=1)  # header="X,Y" 제외
print("CSV shape:", csv_points.shape)
print("CSV 예시 5개:\n", csv_points[:5])

# ------------------------------
# 2단계: NPY 불러오기
# ------------------------------
npy_points = np.load("topview_points.npy")
print("NPY shape:", npy_points.shape)
print("NPY 예시 5개:\n", npy_points[:5])

# ------------------------------
# 3단계: 데이터 일관성 검증
# ------------------------------
if np.allclose(csv_points, npy_points):
    print("✅ CSV와 NPY 파일의 좌표가 완벽히 일치합니다.")
else:
    print("⚠️ CSV와 NPY 간에 일부 값 차이가 존재합니다. 저장 과정 점검 필요.")

# ------------------------------
# 4단계: 시각화 (탑뷰)
# ------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(csv_points[:, 0], csv_points[:, 1],
            c=csv_points[:, 1], cmap="plasma", s=1)

plt.xlabel("Camera X (m, right)")
plt.ylabel("Camera Y (m, forward)")
plt.title("Loaded Top-View Projection (from CSV)")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(-10, 10)
plt.ylim(0, 15)
plt.tight_layout()
plt.show()
