import cv2
import numpy as np
from PIL import Image

def parse_label_file(label_path, img_width, img_height):
    """解析YOLO格式的标签文件，返回像素坐标列表"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    cells = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        # 提取多边形顶点坐标（成对出现）
        coords = list(map(float, parts[1:]))
        # 转换为像素坐标
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i+1] * img_height
            pixel_coords.append((x, y))
        cells.append({
            'class_id': class_id,
            'polygon': pixel_coords,
            'bbox': get_bounding_box(pixel_coords)  # 获取外接矩形
        })
    return cells

def get_bounding_box(polygon):
    """获取多边形的最小外接矩形"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))