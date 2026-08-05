# ============================================
# TrOCR Transformer 
# 
# ============================================

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設定路徑
IMAGE_DIR = r"C:\OCR-AI\src\img"
OUTPUT_DIR = r"C:\OCR-AI\src\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# 1. 營業日報表專用解析器
# ============================================

class DailyReportTrOCRParser:
    """
    營業日報表 TrOCR 解析器
    自動生成日期 (1日, 2日, 3日...)
    """
    
    def __init__(self):
        """初始化解析器，設定表頭結構"""
        
        # 硬編碼的表頭（不包含日期列，日期自動生成）
        self.table_headers = [
            "現金（機1表）", 
            "現金（機2表）", 
            "現金（機3表）", 
            "合計現金（POS）", 
            "差額", 
            "八逹通", 
            "信用卡", 
            "QR code", 
            "Foodpanda", 
            "禮券", 
            "合計營業額", 
            "營業實收", 
            "差額"
        ]
        
        # 需要跳過的列（不匯出）
        self.skip_columns = ["經手人簽署"]
        
        # 數值型欄位（需要轉換為數字）
        self.numeric_columns = [
            "現金（機1表）", "現金（機2表）", "現金（機3表）",
            "合計現金（POS）", "差額", "八逹通", "信用卡",
            "QR code", "Foodpanda", "禮券", "合計營業額",
            "營業實收", "差額"
        ]
        
        # 初始化 TrOCR 模型
        self._init_trocr()
    
    def _init_trocr(self):
        """初始化 TrOCR 模型"""
        print("="*60)
        print("載入 TrOCR 模型...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        
        # 使用手寫模型（適合手寫表格）
        model_name = "microsoft/trocr-base-handwritten"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        print("✓ TrOCR 載入完成")
        print("="*60)
    
    def preprocess_image(self, image_path):
        """圖片前處理"""
        image = Image.open(image_path).convert("RGB")
        
        # 增強對比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # 增強銳利度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    def extract_text_from_image(self, image_path):
        """從圖片提取文字"""
        print(f"\n正在處理: {os.path.basename(image_path)}")
        
        try:
            # 前處理
            image = self.preprocess_image(image_path)
            
            # 轉換為模型輸入
            pixel_values = self.processor(
                images=image, 
                return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # 生成文字
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=512,
                    num_beams=4,
                    temperature=0.5,
                    repetition_penalty=1.2
                )
            
            # 解碼文字
            text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            print(f"✓ 辨識完成，共 {len(text)} 個字元")
            return text
            
        except Exception as e:
            print(f"✗ 錯誤: {e}")
            return ""
    
    def extract_table_by_grid(self, image_path, num_rows=35, num_cols=15):
        """
        使用網格方式逐格辨識表格
        
        Args:
            image_path: 圖片路徑
            num_rows: 行數（包含表頭）
            num_cols: 列數
        
        Returns:
            list: 二維表格資料
        """
        print(f"\n開始逐格辨識表格 ({num_rows} x {num_cols})...")
        
        # 讀取圖片
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # 計算格子大小
        cell_width = width // num_cols
        cell_height = height // num_rows
        
        table_data = []
        total_cells = num_rows * num_cols
        processed = 0
        
        for row in range(num_rows):
            row_data = []
            for col in range(num_cols):
                # 切割單格
                left = col * cell_width
                top = row * cell_height
                right = (col + 1) * cell_width
                bottom = (row + 1) * cell_height
                
                cell_image = image.crop((left, top, right, bottom))
                
                # 辨識單格
                try:
                    pixel_values = self.processor(
                        images=cell_image, 
                        return_tensors="pt"
                    ).pixel_values
                    pixel_values = pixel_values.to(self.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            pixel_values,
                            max_length=20,
                            num_beams=2
                        )
                    
                    text = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0].strip()
                    
                except:
                    text = ""
                
                row_data.append(text)
                processed += 1
                
                # 顯示進度
                if processed % 20 == 0:
                    print(f"  進度: {processed}/{total_cells} ({processed/total_cells*100:.1f}%)")
            
            table_data.append(row_data)
        
        print(f"✓ 表格辨識完成")
        return table_data
    
    def clean_numeric_value(self, value):
        """
        清理數值，提取數字
        
        Args:
            value: 原始文字
        
        Returns:
            float or str: 清理後的數值
        """
        if not value or value.strip() == '':
            return ''
        
        # 移除逗號和空格
        cleaned = str(value).replace(',', '').replace(' ', '')
        
        # 提取數字（包含小數點）
        match = re.search(r'[\d.]+', cleaned)
        if match:
            try:
                return float(match.group())
            except:
                return match.group()
        
        return value
    
    def parse_table_to_dataframe(self, table_data):
        """
        將表格資料轉換為 DataFrame
        
        Args:
            table_data: 二維表格資料
        
        Returns:
            pd.DataFrame: 結構化資料
        """
        if not table_data or len(table_data) < 2:
            print("表格資料不足")
            return pd.DataFrame()
        
        print("\n正在解析表格資料...")
        
        # 跳過表頭行（第一行），從第二行開始是數據
        # 但我們使用硬編碼的表頭，所以只需要數據行
        
        # 找出數據行（跳過空白行和表頭）
        data_rows = []
        
        # 從第1行開始（跳過表頭）
        for row_idx in range(1, len(table_data)):
            row = table_data[row_idx]
            
            # 檢查是否為空行
            if all(cell.strip() == '' for cell in row):
                continue
            
            # 嘗試提取日期（從第一列）
            date_str = row[0] if len(row) > 0 else ''
            
            # 如果第一列包含"日"，則為日期行
            if '日' in date_str or '日' in str(date_str):
                # 提取數字（日期）
                date_match = re.search(r'(\d+)', str(date_str))
                if date_match:
                    day = date_match.group(1)
                    row_data = {
                        '日期': f"{day}日"
                    }
                    
                    # 提取數值欄位
                    for i, header in enumerate(self.table_headers):
                        # +1 因為第一列是日期
                        col_idx = i + 1
                        if col_idx < len(row):
                            value = self.clean_numeric_value(row[col_idx])
                            row_data[header] = value
                        else:
                            row_data[header] = ''
                    
                    data_rows.append(row_data)
        
        # 建立 DataFrame
        df = pd.DataFrame(data_rows)
        
        # 如果沒有找到日期，自動生成日期
        if len(df) == 0:
            print("未找到日期資料，自動生成日期...")
            return self.generate_dates_from_data(table_data)
        
        print(f"✓ 解析完成，共 {len(df)} 行")
        return df
    
    def generate_dates_from_data(self, table_data):
        """
        如果 OCR 無法辨識日期，自動生成日期
        
        Args:
            table_data: 原始表格資料
        
        Returns:
            pd.DataFrame: 包含自動生成日期的 DataFrame
        """
        print("自動生成日期 (1日, 2日, 3日...)")
        
        data_rows = []
        day_counter = 1
        
        for row_idx in range(1, len(table_data)):
            row = table_data[row_idx]
            
            # 檢查是否為空行
            if all(cell.strip() == '' for cell in row):
                continue
            
            # 檢查是否有任何數值
            has_value = False
            for cell in row[1:]:  # 跳過第一列
                if cell.strip() and re.search(r'[\d.]+', cell):
                    has_value = True
                    break
            
            if has_value:
                row_data = {
                    '日期': f"{day_counter}日"
                }
                
                # 提取數值欄位
                for i, header in enumerate(self.table_headers):
                    col_idx = i + 1
                    if col_idx < len(row):
                        value = self.clean_numeric_value(row[col_idx])
                        row_data[header] = value
                    else:
                        row_data[header] = ''
                
                data_rows.append(row_data)
                day_counter += 1
        
        df = pd.DataFrame(data_rows)
        print(f"✓ 自動生成 {len(df)} 行")
        return df
    
    def process_image(self, image_path, use_grid=True):
        """
        處理單張圖片
        
        Args:
            image_path: 圖片路徑
            use_grid: 是否使用網格方式
        
        Returns:
            pd.DataFrame: 解析結果
        """
        if use_grid:
            # 網格方式（較準確）
            table_data = self.extract_table_by_grid(
                image_path, 
                num_rows=35, 
                num_cols=15
            )
            df = self.parse_table_to_dataframe(table_data)
        else:
            # 整張圖片辨識（較快速）
            text = self.extract_text_from_image(image_path)
            # 解析文字...
            df = self.parse_text_to_dataframe(text)
        
        # 清理並轉換數值
        df = self.clean_dataframe(df)
        
        # 儲存結果
        self.save_results(df, image_path)
        
        return df
    
    def parse_text_to_dataframe(self, text):
        """從純文字解析表格"""
        lines = text.strip().split('\n')
        
        data_rows = []
        day_counter = 1
        
        for line in lines:
            # 嘗試分割
            cells = re.split(r'\s{2,}', line.strip())
            cells = [c.strip() for c in cells if c.strip()]
            
            if len(cells) >= 2:
                # 檢查是否包含日期
                date_match = re.search(r'(\d+)\s*日', cells[0])
                if date_match:
                    day = date_match.group(1)
                else:
                    day = str(day_counter)
                    day_counter += 1
                
                row_data = {'日期': f"{day}日"}
                
                for i, header in enumerate(self.table_headers):
                    if i + 1 < len(cells):
                        row_data[header] = self.clean_numeric_value(cells[i + 1])
                    else:
                        row_data[header] = ''
                
                data_rows.append(row_data)
        
        return pd.DataFrame(data_rows)
    
    def clean_dataframe(self, df):
        """清理 DataFrame，轉換數值型別"""
        if df.empty:
            return df
        
        # 確保日期列存在
        if '日期' not in df.columns:
            df.insert(0, '日期', [f"{i+1}日" for i in range(len(df))])
        
        # 轉換數值欄位
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 過濾跳過的列
        for col in self.skip_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        return df
    
    def save_results(self, df, image_path):
        """儲存結果"""
        if df.empty:
            print("沒有資料可儲存")
            return
        
        # 建立輸出檔名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 儲存為 CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_daily_report.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV 已儲存: {csv_path}")
        
        # 儲存為 Excel（可選）
        try:
            excel_path = os.path.join(OUTPUT_DIR, f"{base_name}_daily_report.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"✓ Excel 已儲存: {excel_path}")
        except:
            pass
        
        # 顯示摘要
        print("\n" + "="*60)
        print("結果摘要")
        print("="*60)
        print(f"總行數: {len(df)}")
        print(f"總列數: {len(df.columns)}")
        print(f"日期範圍: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")
        
        # 顯示前幾行
        print("\n前 5 行:")
        print(df.head())
    
    def process_all_images(self):
        """批次處理所有圖片"""
        image_files = [
            f for f in os.listdir(IMAGE_DIR)
            if f.endswith(('.jpeg', '.jpg', '.png'))
        ]
        
        if not image_files:
            print("未找到圖片檔案")
            return
        
        print(f"找到 {len(image_files)} 張圖片")
        
        all_dfs = []
        for image_file in image_files:
            image_path = os.path.join(IMAGE_DIR, image_file)
            df = self.process_image(image_path, use_grid=True)
            if not df.empty:
                df['來源'] = image_file
                all_dfs.append(df)
        
        # 合併所有結果
        if all_dfs and len(all_dfs) > 1:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_path = os.path.join(OUTPUT_DIR, "combined_daily_reports.csv")
            combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 合併檔案已儲存: {combined_path}")
        
        return all_dfs


# ============================================
# 2. 主程式執行
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("TrOCR 營業日報表解析系統")
    print("="*60)
    print(f"圖片目錄: {IMAGE_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print("="*60)
    
    # 初始化解析器
    parser = DailyReportTrOCRParser()
    
    # 處理所有圖片
    results = parser.process_all_images()
    
    print("\n" + "="*60)
    print("所有處理完成！")
    print("="*60)