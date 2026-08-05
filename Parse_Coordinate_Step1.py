import cv2
import numpy as np
from PIL import Image

def parse_label_file(label_path, img_width, img_height):
    """解析YOLO格式的标签文件，返回像素坐标列表"""
    with open(label_path, 'r') as f:  # 改为使用传入的参数
        lines = f.readlines()
    
    cells = []
    for line in lines:
        parts = line.strip().split()
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

# ---------- 主程序入口 ----------
if __name__ == "__main__":
    # 设置正确的路径
    label_path = r"C:\OGHFYOLO\runs\predict-seg\exp7\labels\data_Image1- clear.txt"
    image_path = r"C:\OGHFYOLO\runs\predict-seg\exp7\data_Image1- clear.jpeg"
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        exit(1)
    h, w = img.shape[:2]
    print(f"图片尺寸: {w} x {h}")
    
    # 解析标签
    cells = parse_label_file(label_path, w, h)
    print(f"共识别出 {len(cells)} 个单元格")
    
    # 打印前3个单元格信息
    for i, cell in enumerate(cells[:3]):
        bbox = cell['bbox']
        print(f"\n单元格 {i+1}:")
        print(f"  外接矩形: ({bbox[0]:.1f}, {bbox[1]:.1f}) -> ({bbox[2]:.1f}, {bbox[3]:.1f})")
        print(f"  多边形顶点数: {len(cell['polygon'])}")
    
    # 保存第一个单元格的轮廓图
    if cells:
        img_copy = img.copy()
        pts = np.array(cells[0]['polygon'], dtype=np.int32)
        cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
        cv2.imwrite("test_cell_contour.jpg", img_copy)
        print("\n已保存第一个单元格的轮廓图: test_cell_contour.jpg")