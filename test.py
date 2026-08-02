import os
import re
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract


# 指定 Tesseract 可执行文件的路径
tesseract_path = r'C:\OCR-AI\Tesseract\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"Tesseract found at: {tesseract_path}")
else:
    print(f"Warning: Tesseract not found at {tesseract_path}")
    
    
def load_images(directory: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
    return sorted(files)


def resize_to_reasonable(img: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def preprocess_for_table(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 讓細線更清楚，保留表格格線和文字輪廓
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反轉：將線條保留為黑，背景白
    bw = 255 - bw

    # 清除小雜點
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    return bw


def find_table_mask(img_bin: np.ndarray) -> np.ndarray:
    # 找出最大輪廓，通常是整張紙/表格區域
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(img_bin, dtype=np.uint8) * 255

    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_bin)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=-1)
    return mask


def detect_grid_lines(binary: np.ndarray) -> Tuple[List[int], List[int]]:
    h, w = binary.shape
    # 以黑像素為主，先把背景去掉
    binary = np.where(binary > 200, 255, 0).astype(np.uint8)

    # 水平線投影
    row_proj = np.sum(binary == 0, axis=1)
    # 僅保留高於平均的行，定位線條
    row_thresh = max(10, int(np.mean(row_proj[row_proj > 0]) * 0.45))
    row_lines = np.where(row_proj > row_thresh)[0]
    row_bounds = []
    start = None
    for idx in row_lines:
        if start is None:
            start = idx
        elif idx - prev > 1:
            if prev - start >= 1:
                row_bounds.append((start, prev))
            start = idx
        prev = idx
    if start is not None and row_lines[-1] >= start:
        row_bounds.append((start, row_lines[-1]))

    col_proj = np.sum(binary == 0, axis=0)
    col_thresh = max(10, int(np.mean(col_proj[col_proj > 0]) * 0.45))
    col_lines = np.where(col_proj > col_thresh)[0]
    col_bounds = []
    start = None
    for idx in col_lines:
        if start is None:
            start = idx
        elif idx - prev > 1:
            if prev - start >= 1:
                col_bounds.append((start, prev))
            start = idx
        prev = idx
    if start is not None and col_lines[-1] >= start:
        col_bounds.append((start, col_lines[-1]))

    # 如果沒有明確表格線，退回到整體輪廓的中位數分割
    if not row_bounds or not col_bounds:
        # 使用固定間距 fallback：每 1/8 頁寬高取序列
        row_bounds = [(0, h - 1)]
        col_bounds = [(0, w - 1)]

    rows = sorted({int((a + b) / 2) for a, b in row_bounds})
    cols = sorted({int((a + b) / 2) for a, b in col_bounds})
    if not rows:
        rows = [0, h - 1]
    if not cols:
        cols = [0, w - 1]
    return rows, cols


def extract_table_cells(img_bin: np.ndarray) -> List[List[np.ndarray]]:
    mask = find_table_mask(img_bin)
    combined = cv2.bitwise_and(img_bin, mask)
    h, w = combined.shape
    rows, cols = detect_grid_lines(combined)

    # 讓 rows / cols 能形成邊界
    y_lines = [0]
    y_lines.extend(rows)
    y_lines.append(h)
    y_lines = sorted(set(y_lines))

    x_lines = [0]
    x_lines.extend(cols)
    x_lines.append(w)
    x_lines = sorted(set(x_lines))

    cells = []
    for r in range(len(y_lines) - 1):
        row = []
        for c in range(len(x_lines) - 1):
            x0, x1 = x_lines[c], x_lines[c + 1]
            y0, y1 = y_lines[r], y_lines[r + 1]
            if x1 <= x0 or y1 <= y0:
                continue
            cell = combined[y0:y1, x0:x1]
            row.append(cell)
        cells.append(row)

    # 手寫/細線表格常常沒有清楚的 grid，則將整張表當作一格
    if len(cells) == 1 and len(cells[0]) == 1:
        cells = [[combined]]

    return cells


def ocr_cell(cell: np.ndarray) -> str:
    if cell.size == 0:
        return ""

    # 先將白底轉為黑字，方便 OCR
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    if gray.size == 0:
        return ""

    # 轉成 0/255 binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 反轉成黑字白底，OCR 更穩定
    binary = 255 - binary

    # 放大後 OCR
    h, w = binary.shape
    if max(h, w) > 0:
        scale = 2
        binary = cv2.resize(binary, (max(1, w * scale), max(1, h * scale)), interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(binary, config="--psm 11 --oem 3")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def process_image_to_csv(image_path: str, out_dir: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = resize_to_reasonable(img)
    img_bin = preprocess_for_table(img)
    cells = extract_table_cells(img_bin)

    rows_data: List[List[str]] = []
    for row in cells:
        values = []
        for cell in row:
            if isinstance(cell, np.ndarray):
                if cell.ndim == 3:
                    text = ocr_cell(cell)
                else:
                    text = ocr_cell(cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR))
            else:
                text = ""
            values.append(text)
        rows_data.append(values)

    # 補齊矩形表格
    max_cols = max(len(r) for r in rows_data) if rows_data else 1
    for r in rows_data:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    df = pd.DataFrame(rows_data)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}.csv")
    df.to_csv(out_path, index=False, header=False, encoding="utf-8-sig")
    return out_path


def main():
    image_dir = r"C:\OCR-AI\src\img"
    out_dir = r"C:\OCR-AI\src\output"
    os.makedirs(out_dir, exist_ok=True)

    images = load_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")

    print(f"Found {len(images)} image(s):")
    for p in images:
        print(" -", os.path.basename(p))

    for image_path in images:
        out_csv = process_image_to_csv(image_path, out_dir)
        print(f"CSV saved: {out_csv}")


if __name__ == "__main__":
    main()
