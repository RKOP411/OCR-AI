import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ---------------- 1. 加载模型 ----------------
print("🔄 加载 TrOCR 模型...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ 模型加载完成，使用设备: {device}")

# ---------------- 2. 直接读取 debug 图片 ----------------
image_path = r"C:\OCR-AI\debug_crops\row0_col26.jpg"  # 换成你的实际路径
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    print(f"❌ 无法读取图片: {image_path}")
    exit()

print(f"📷 图片尺寸: {img_bgr.shape}")
print(f"🔍 图片类型: {img_bgr.dtype}, 通道数: {img_bgr.shape[2] if len(img_bgr.shape)==3 else 1}")

# ---------------- 3. 关键步骤：颜色与通道处理 ----------------
# 如果你之前代码里没处理好通道，这里做一个最稳妥的转换
if len(img_bgr.shape) == 2:
    # 如果是灰度图，转成 3 通道 RGB (TrOCR 需要 3 通道)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
else:
    # 如果是 BGR，转成 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 转换为 PIL Image
pil_image = Image.fromarray(img_rgb)

# ---------------- 4. 执行识别 ----------------
try:
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    generated_ids = model.generate(pixel_values, max_length=64)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print("-" * 30)
    print(f"✅ TrOCR 最终识别结果: '{text}'")
    print("-" * 30)
    
except Exception as e:
    print(f"❌ 识别过程中发生错误: {e}")