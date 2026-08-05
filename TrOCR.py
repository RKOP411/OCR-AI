from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 加载TrOCR模型
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def recognize_with_trocr(image):
    """使用TrOCR识别单个单元格"""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text