import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

def parse_label_file(label_path, img_width, img_height):
    """解析YOLO格式的标签文件，返回像素坐标列表"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    cells = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:  # 跳过空行或无效行
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
    """根据外接矩形裁剪单元格，添加少量边距"""
    x_min, y_min, x_max, y_max = bbox
    # 添加边距，防止裁剪太紧导致文字被切
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(image.shape[1], int(x_max) + padding)
    y_max = min(image.shape[0], int(y_max) + padding)
    return image[y_min:y_max, x_min:x_max]

def get_cell_center(cell):
    """获取单元格中心点坐标（用于排序）"""
    bbox = cell['bbox']
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def sort_cells_by_position(cells, row_threshold=0.3):
    """
    按单元格位置排序（从上到下，从左到右）
    row_threshold: 行间距的阈值比例（相对于图片高度）
    """
    if not cells:
        return []
    
    h = 1.0  # 归一化高度，用于计算阈值
    # 获取每个单元格的中心
    for cell in cells:
        center_x, center_y = get_cell_center(cell)
        cell['center_x'] = center_x
        cell['center_y'] = center_y
    
    # 按Y坐标排序（从上到下）
    sorted_by_y = sorted(cells, key=lambda c: c['center_y'])
    
    # 根据Y坐标分组为行
    rows = []
    current_row = [sorted_by_y[0]]
    current_y = sorted_by_y[0]['center_y']
    
    # 计算行间距阈值（图片高度的百分比）
    threshold = h * 0.05  # 5%的图片高度作为行间距阈值
    
    for cell in sorted_by_y[1:]:
        if abs(cell['center_y'] - current_y) > threshold:
            # 新行，对当前行按X排序
            current_row.sort(key=lambda c: c['center_x'])
            rows.append(current_row)
            current_row = [cell]
            current_y = cell['center_y']
        else:
            current_row.append(cell)
    
    # 处理最后一行
    if current_row:
        current_row.sort(key=lambda c: c['center_x'])
        rows.append(current_row)
    
    return rows

def recognize_with_trocr(image):
    """
    使用TrOCR识别单个单元格图像
    这里先用占位符，后续接入真实模型
    """
    # 占位：返回空字符串，实际使用时替换为TrOCR调用
    # 示例：
    # from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    # model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    # pixel_values = processor(images=image, return_tensors="pt").pixel_values
    # generated_ids = model.generate(pixel_values)
    # text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # return text
    return "[待识别]"

def process_table_image(image_path, label_path, output_csv="table_output.csv", debug=False):
    """
    处理单张表格图片的完整流程
    """
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
    rows = sort_cells_by_position(cells)
    print(f"📋 按位置划分为 {len(rows)} 行")
    
    # 4. 裁剪并识别每个单元格
    table_data = []
    for row_idx, row in enumerate(rows):
        row_data = []
        for col_idx, cell in enumerate(row):
            # 裁剪单元格
            crop_img = crop_cell(img, cell['bbox'], padding=2)
            
            # 调试：保存裁剪的单元格图像
            if debug:
                debug_dir = "debug_crops"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/row{row_idx}_col{col_idx}.jpg", crop_img)
            
            # 使用TrOCR识别（目前是占位符）
            text = recognize_with_trocr(crop_img)
            
            # 提取外接矩形坐标（用于定位）
            bbox = cell['bbox']
            
            row_data.append({
                'row': row_idx,
                'col': col_idx,
                'text': text,
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3],
                'center_x': cell.get('center_x', 0),
                'center_y': cell.get('center_y', 0)
            })
        table_data.extend(row_data)
    
    # 5. 转换为DataFrame
    df = pd.DataFrame(table_data)
    
    # 6. 保存CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ 数据已保存至: {output_csv}")
    
    # 7. 打印预览
    print("\n📄 表格预览 (前5行):")
    print(df[['row', 'col', 'text']].head())
    
    return df

# ---------- 主程序入口 ----------
if __name__ == "__main__":
    # 设置路径（根据你的实际路径修改）
    label_path = r"C:\OGHFYOLO\runs\predict-seg\exp7\labels\data_Image1- clear.txt"
    image_path = r"C:\OGHFYOLO\runs\predict-seg\exp7\data_Image1- clear.jpeg"
    
    # 执行处理
    df = process_table_image(
        image_path=image_path,
        label_path=label_path,
        output_csv="table_output.csv",
        debug=True  # 开启调试模式，会保存裁剪的单元格图片
    )
    
    if df is not None:
        print(f"\n📊 共处理 {len(df)} 个单元格")
        print(f"📊 表格结构: {df['row'].max()+1} 行 x {df['col'].max()+1} 列")