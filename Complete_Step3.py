import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ---------- 加载TrOCR模型 ----------
print("🔄 加载TrOCR模型...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ TrOCR加载完成，使用设备: {device}")

# ---------- 坐标解析函数 ----------
def parse_label_file(label_path, img_width, img_height):
    """解析YOLO格式的标签文件，返回像素坐标列表"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    cells = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i+1] * img_height
            pixel_coords.append((x, y))
        cells.append({
            'class_id': class_id,
            'polygon': pixel_coords,
            'bbox': get_bounding_box(pixel_coords)
        })
    return cells

def get_bounding_box(polygon):
    """获取多边形的最小外接矩形"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))

def crop_cell(image, bbox, padding=5):
    """根据外接矩形裁剪单元格"""
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(image.shape[1], int(x_max) + padding)
    y_max = min(image.shape[0], int(y_max) + padding)
    return image[y_min:y_max, x_min:x_max]

def get_cell_center(cell):
    """获取单元格中心点坐标"""
    bbox = cell['bbox']
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def sort_cells_by_position(cells, h):
    """按位置排序单元格（从上到下，从左到右）"""
    if not cells:
        return []
    
    for cell in cells:
        center_x, center_y = get_cell_center(cell)
        cell['center_x'] = center_x
        cell['center_y'] = center_y
    
    sorted_by_y = sorted(cells, key=lambda c: c['center_y'])
    
    # 自适应阈值
    diffs = []
    for i in range(1, len(sorted_by_y)):
        diff = sorted_by_y[i]['center_y'] - sorted_by_y[i-1]['center_y']
        diffs.append(diff)
    
    if not diffs:
        return [sorted_by_y]
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    adaptive_threshold = max(mean_diff + std_diff * 0.5, 30)
    
    rows = []
    current_row = [sorted_by_y[0]]
    
    for cell in sorted_by_y[1:]:
        y_diff = cell['center_y'] - current_row[-1]['center_y']
        if y_diff > adaptive_threshold:
            current_row.sort(key=lambda c: c['center_x'])
            rows.append(current_row)
            current_row = [cell]
        else:
            current_row.append(cell)
    
    if current_row:
        current_row.sort(key=lambda c: c['center_x'])
        rows.append(current_row)
    
    return rows

# ---------- TrOCR识别函数 ----------
def recognize_with_trocr(image):
    """使用TrOCR识别单个单元格图像"""
    try:
        # 转换为PIL图像（TrOCR需要）
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
        
        # 处理并生成
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        generated_ids = model.generate(pixel_values, max_length=64)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 清理空白字符
        text = text.strip()
        return text if text else ""
    except Exception as e:
        print(f"   ⚠️ TrOCR识别错误: {e}")
        return ""

# ---------- 主处理函数 ----------
def process_table_image(image_path, label_path, output_csv="table_output.csv", debug=False):
    """处理单张表格图片的完整流程"""
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    h, w = img.shape[:2]
    print(f"📷 图片尺寸: {w} x {h}")
    
    # 2. 解析标签文件
    cells = parse_label_file(label_path, w, h)
    print(f"📊 识别出 {len(cells)} 个单元格")
    
    # 3. 按位置排序
    rows = sort_cells_by_position(cells, h)
    print(f"📋 按位置划分为 {len(rows)} 行")
    
    # 4. 裁剪并识别每个单元格
    table_data = []
    total_cells = sum(len(row) for row in rows)
    processed = 0
    
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            # 裁剪单元格
            crop_img = crop_cell(img, cell['bbox'], padding=2)
            
            # 调试：保存裁剪的单元格图像
            if debug:
                debug_dir = "debug_crops"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/row{row_idx}_col{col_idx}.jpg", crop_img)
            
            # TrOCR识别
            text = recognize_with_trocr(crop_img)
            
            # 进度显示
            processed += 1
            if processed % 50 == 0 or processed == total_cells:
                print(f"   进度: {processed}/{total_cells}")
            
            bbox = cell['bbox']
            table_data.append({
                'row': row_idx,
                'col': col_idx,
                'text': text,
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3]
            })
    
    # 5. 转换为DataFrame并保存
    df = pd.DataFrame(table_data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 数据已保存至: {output_csv}")
    
    # 6. 打印预览
    print("\n📄 表格预览 (前10行):")
    print(df[['row', 'col', 'text']].head(10))
    
    return df

# ---------- 主程序入口 ----------
if __name__ == "__main__":
    label_path = r"C:\OGHFYOLO\runs\predict-seg\exp7\labels\data_Image1- clear.txt"
    image_path = r"C:\OCR-AI\src\img\data_Image1.jpeg"
    
    df = process_table_image(
        image_path=image_path,
        label_path=label_path,
        output_csv="table_output.csv",
        debug=True
    )
    
    if df is not None:
        print(f"\n📊 表格结构: {df['row'].max()+1} 行 x {df['col'].max()+1} 列")
        print(f"📊 非空单元格: {df[df['text'] != ''].shape[0]}/{len(df)}")