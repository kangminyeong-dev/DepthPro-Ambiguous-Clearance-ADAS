# debug01.py
import cv2
import torch
import depth_pro
import matplotlib.pyplot as plt

# ======================================================
# 단일 테스트 이미지 경로
# ======================================================
img_path = "test_frames1/01.png"
print(f"[DEBUG] 입력 이미지: {img_path}")

# ======================================================
# 디바이스 & FP16 설정
# ======================================================
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = (device.type == "cuda")
print(f"Device: {device}, FP16: {use_fp16}")

# ======================================================
# DepthPro 로드
# ======================================================
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device)
if use_fp16:
    model = model.half()
model.eval()

# ======================================================
# DepthPro 추론
# ======================================================
img, _, f_px = depth_pro.load_rgb(img_path)
image_t = transform(img).to(device)
if use_fp16:
    image_t = image_t.half()

with torch.no_grad():
    prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"].squeeze().detach().cpu().numpy()

print("[DEBUG] Depth shape:", depth.shape)

# ======================================================
# 원본 이미지 시각화 출력
# ======================================================
orig_bgr = cv2.imread(img_path)
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6), dpi=500)
plt.imshow(orig_rgb)
plt.axis("off")
plt.title("Original Image: 01.png")
plt.show()
