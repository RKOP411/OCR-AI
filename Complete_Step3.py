import os
from PIL import Image
import pandas as pd

def process_table_image(image_path, label_path):
    """处理单张表格图片，输出CSV数据"""
    # 1. 获取图片尺寸
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 2. 解析标签
    cells = parse_label_file(label_path, w, h)
    
    # 3. 对每个单元格进行处理
    results = []
    for i, cell in enumerate(cells):
        bbox = cell['bbox']
        # 裁剪单元格
        crop = crop_cell_from_image(image_path, bbox)
        
        # 4. 送入TrOCR识别
        # text = your_trocr_model.recognize(crop)
        text = "待识别"  # 替换为实际TrOCR调用
        
        # 记录位置信息（用于排序）
        results.append({
            'row': i,  # 需要根据y坐标排序来确定行
            'col': i,  # 需要根据x坐标排序来确定列
            'x_center': (bbox[0] + bbox[2]) / 2,
            'y_center': (bbox[1] + bbox[3]) / 2,
            'text': text,
            'confidence': 0.9  # TrOCR的置信度
        })
    
    # 5. 按位置排序
    results = sort_cells_by_position(results)
    
    # 6. 导出CSV
    df = pd.DataFrame(results)
    df.to_csv('table_output.csv', index=False)
    return df

def sort_cells_by_position(cells):
    """按单元格位置排序（从左到右，从上到下）"""
    # 先按y排序分出行，再按x排序分出列
    sorted_cells = sorted(cells, key=lambda c: (c['y_center'], c['x_center']))
    
    # 根据y坐标的分布自动分组为行
    rows = []
    current_row = []
    current_y = sorted_cells[0]['y_center'] if sorted_cells else 0
    threshold = 20  # 像素阈值，根据实际情况调整
    
    for cell in sorted_cells:
        if abs(cell['y_center'] - current_y) > threshold:
            rows.append(sorted(current_row, key=lambda c: c['x_center']))
            current_row = [cell]
            current_y = cell['y_center']
        else:
            current_row.append(cell)
    if current_row:
        rows.append(sorted(current_row, key=lambda c: c['x_center']))
    
    return rows