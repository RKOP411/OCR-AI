# ocr_main.py - 自动生成日期列
import os
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import pytesseract
import threading
import datetime
import json
import re

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR 表格数字识别工具")
        self.root.geometry("1000x700")
        
        # 设置Tesseract路径
        self.setup_tesseract()
        
        # 创建UI
        self.create_widgets()
        
        # 当前状态
        self.current_image = None
        self.current_image_path = None
        self.history = self.load_history()
        self.is_processing = False
        
        # 设置拖拽
        self.setup_drag_drop()
        
        # 硬编码的表头（不包含日期列，日期自动生成）
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
        
        # 需要跳过的列
        self.skip_columns = ["經手人簽署"]
    
    def setup_tesseract(self):
        """配置Tesseract路径"""
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        tesseract_path = os.path.join(base_path, 'Tesseract', 'tesseract.exe')
        
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✅ 使用便携版Tesseract: {tesseract_path}")
        else:
            try:
                pytesseract.get_tesseract_version()
                print("✅ 使用系统安装的Tesseract")
            except:
                import subprocess
                try:
                    result = subprocess.run(['where', 'tesseract'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        tesseract_path = result.stdout.strip().split('\n')[0]
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                        print(f"✅ 在PATH中找到Tesseract: {tesseract_path}")
                    else:
                        raise Exception("Tesseract not found")
                except:
                    messagebox.showerror("错误", 
                        "找不到Tesseract！\n"
                        "请确保Tesseract文件夹在程序目录下，\n"
                        "或者安装Tesseract到系统。")
                    sys.exit(1)
    
    def setup_drag_drop(self):
        """设置拖拽功能"""
        try:
            self.root.drop_target_register('DND_Files')
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        except:
            print("拖拽功能不可用")
    
    def on_drop(self, event):
        """处理拖拽文件"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                self.load_image(file_path)
    
    def create_widgets(self):
        """创建UI组件"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 工具栏
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        btn_config = {'padx': 10, 'pady': 5, 'relief': tk.RAISED, 'bd': 1}
        
        btn_frame1 = tk.Frame(toolbar)
        btn_frame1.pack(side=tk.LEFT, fill=tk.X)
        
        tk.Button(btn_frame1, text="📁 选择图片", command=self.select_image, **btn_config).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame1, text="📂 选择文件夹", command=self.select_folder, **btn_config).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame1, text="🔍 识别", command=self.start_ocr, **btn_config).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame1, text="💾 保存", command=self.save_result, **btn_config).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame1, text="📋 复制", command=self.copy_result, **btn_config).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame1, text="🧹 清空", command=self.clear_all, **btn_config).pack(side=tk.LEFT, padx=2)
        
        # 设置区域
        settings_frame = tk.Frame(toolbar)
        settings_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        # 识别模式选择
        tk.Label(settings_frame, text="识别模式:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="表格数字")
        mode_menu = ttk.Combobox(settings_frame, textvariable=self.mode_var,
                                values=["表格数字", "普通文本", "仅数字"],
                                width=12)
        mode_menu.pack(side=tk.LEFT, padx=5)
        
        # 语言选择
        tk.Label(settings_frame, text="语言:").pack(side=tk.LEFT, padx=(10, 0))
        self.lang_var = tk.StringVar(value="chi_sim+eng")
        lang_menu = ttk.Combobox(settings_frame, textvariable=self.lang_var,
                                values=["eng", "chi_sim", "chi_sim+eng", "jpn", "kor"],
                                width=12)
        lang_menu.pack(side=tk.LEFT, padx=5)
        
        # 图片预处理选项
        tk.Label(settings_frame, text="预处理:").pack(side=tk.LEFT, padx=(10, 0))
        self.preprocess_var = tk.StringVar(value="增强对比度")
        preprocess_menu = ttk.Combobox(settings_frame, textvariable=self.preprocess_var,
                                     values=["无", "灰度", "二值化", "降噪", "增强对比度", "锐化"],
                                     width=10)
        preprocess_menu.pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, 
                                           maximum=100, length=150)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # 主区域
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左半部分 - 图片显示
        left_frame = tk.Frame(main_paned)
        main_paned.add(left_frame, width=400)
        
        # 图片信息
        info_frame = tk.Frame(left_frame)
        info_frame.pack(fill=tk.X, pady=2)
        
        self.image_info = tk.Label(info_frame, text="图片信息: 未加载", anchor=tk.W)
        self.image_info.pack(side=tk.LEFT)
        
        tk.Button(info_frame, text="放大", command=self.zoom_in).pack(side=tk.RIGHT, padx=2)
        tk.Button(info_frame, text="缩小", command=self.zoom_out).pack(side=tk.RIGHT, padx=2)
        
        # 图片显示
        canvas_frame = tk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='#f0f0f0')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar_v = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_h = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(xscrollcommand=scrollbar_h.set, yscrollcommand=scrollbar_v.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        self.image_on_canvas = None
        self.zoom_level = 1.0
        
        # 右半部分 - 文字结果
        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, width=500)
        
        # 结果标签
        result_header = tk.Frame(right_frame)
        result_header.pack(fill=tk.X)
        
        tk.Label(result_header, text="识别结果", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        self.word_count_label = tk.Label(result_header, text="数据行: 0", fg='gray')
        self.word_count_label.pack(side=tk.RIGHT)
        
        # 结果显示
        self.result_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                                    font=('Consolas', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.bind('<KeyRelease>', self.update_word_count)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_canvas_configure(self, event):
        if self.image_on_canvas:
            self.display_image_on_canvas()
    
    def display_image_on_canvas(self):
        """在画布上显示图片"""
        if not self.current_image:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        img_width, img_height = self.current_image.size
        img_width *= self.zoom_level
        img_height *= self.zoom_level
        
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)
        
        img_copy = self.current_image.copy()
        if self.zoom_level != 1.0:
            new_size = (int(img_copy.width * self.zoom_level), 
                       int(img_copy.height * self.zoom_level))
            img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)
        
        # 预览预处理
        img_copy = self.apply_preprocessing(img_copy, preview=True)
        
        photo = ImageTk.PhotoImage(img_copy)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                image=photo, anchor=tk.CENTER)
        self.image_on_canvas = photo
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def apply_preprocessing(self, image, preview=False):
        """应用图片预处理"""
        preprocess = self.preprocess_var.get()
        
        # 如果是预览，只做简单的处理
        if preview:
            if preprocess == "灰度":
                if image.mode != 'L':
                    image = image.convert('L')
            return image
        
        # 完整预处理（用于OCR）
        # 1. 转换为灰度
        if image.mode != 'L':
            image = image.convert('L')
        
        # 2. 增强对比度
        if preprocess in ["增强对比度", "锐化"]:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.5)
        
        # 3. 降噪
        if preprocess in ["降噪", "增强对比度"]:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # 4. 二值化
        if preprocess == "二值化":
            threshold = 150
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
        
        # 5. 锐化
        if preprocess == "锐化":
            image = image.filter(ImageFilter.SHARPEN)
        
        # 6. 放大图片（提高OCR准确率）
        width, height = image.size
        if width < 1000 or height < 1000:
            scale = max(2, min(2000/width, 2000/height))
            if scale > 1:
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def zoom_in(self):
        self.zoom_level *= 1.2
        self.display_image_on_canvas()
    
    def zoom_out(self):
        self.zoom_level /= 1.2
        if self.zoom_level < 0.1:
            self.zoom_level = 0.1
        self.display_image_on_canvas()
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if file_path:
            self.load_image(file_path)
    
    def select_folder(self):
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        if folder_path:
            self.batch_process(folder_path)
    
    def load_image(self, image_path):
        try:
            self.current_image_path = image_path
            self.current_image = Image.open(image_path)
            self.zoom_level = 1.0
            
            size = self.current_image.size
            mode = self.current_image.mode
            self.image_info.config(text=f"图片: {os.path.basename(image_path)} | 尺寸: {size[0]}x{size[1]} | 模式: {mode}")
            
            self.display_image_on_canvas()
            self.status_bar.config(text=f"已加载: {os.path.basename(image_path)}")
            
            self.start_ocr()
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
    def start_ocr(self):
        if not self.current_image:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        threading.Thread(target=self.do_ocr, daemon=True).start()
    
    def do_ocr(self):
        """执行OCR识别"""
        try:
            self.root.after(0, lambda: self.status_bar.config(text="正在识别..."))
            self.root.after(0, lambda: self.root.config(cursor="watch"))
            self.root.after(0, lambda: self.progress_bar.start(10))
            
            # 获取语言
            lang = self.lang_var.get()
            mode = self.mode_var.get()
            
            # 应用预处理
            processed_image = self.apply_preprocessing(self.current_image)
            
            # 设置Tesseract参数 - 优化数字识别
            custom_config = r'--oem 3 --psm 6'
            
            # 根据模式调整参数
            if mode == "仅数字":
                custom_config += r' -c tessedit_char_whitelist=0123456789.,-'
                lang = "eng"
            elif mode == "表格数字":
                # 允许数字、小数点、负号
                custom_config += r' -c tessedit_char_whitelist=0123456789.,-'
                lang = "eng"
            
            # 执行OCR - 只识别数字
            text = pytesseract.image_to_string(processed_image, lang=lang, config=custom_config)
            
            # 根据模式处理结果
            if mode == "表格数字":
                text = self.process_table_data(text)
            elif mode == "仅数字":
                text = self.extract_numbers(text)
            
            self.root.after(0, lambda: self.update_result(text))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("OCR错误", str(e)))
        finally:
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.status_bar.config(text="识别完成"))
            self.root.after(0, lambda: self.root.config(cursor=""))
            self.is_processing = False
    
    def process_table_data(self, text):
        """处理表格数据，按行分组数字"""
        # 提取所有数字（包括小数）
        all_numbers = re.findall(r'(\d+\.?\d*)', text)
        
        if not all_numbers:
            return "未识别到数字数据"
        
        # 每行应该有13列数据（对应13个表头）
        columns_per_row = len(self.table_headers)
        
        # 按行分组
        rows = []
        current_row = []
        
        for num in all_numbers:
            current_row.append(num)
            if len(current_row) >= columns_per_row:
                rows.append(current_row)
                current_row = []
        
        # 如果还有剩余数据，也添加为一行
        if current_row:
            rows.append(current_row)
        
        # 格式化输出
        return self.format_table(rows)
    
    def format_table(self, rows):
        """格式化为漂亮的表格"""
        if not rows:
            return "未识别到数据"
        
        # 构建表头（包含日期列）
        headers = ["日期"] + self.table_headers
        
        # 构建表格
        result_lines = []
        
        # 表头
        result_lines.append("| " + " | ".join(headers) + " |")
        result_lines.append("|" + "|".join(["------"] * len(headers)) + "|")
        
        # 数据行 - 自动生成日期（1日到31日）
        for i, row_data in enumerate(rows, 1):
            # 日期从1日开始
            day = i if i <= 31 else i % 31
            date_str = f"{day}日"
            
            # 填充数据，不足的列留空
            full_row = [date_str] + row_data[:len(self.table_headers)]
            
            # 如果数据列不足，补充空值
            while len(full_row) < len(headers):
                full_row.append("")
            
            # 检查是否该行有数据（除了日期外）
            has_data = any(full_row[1:])
            if has_data:
                result_lines.append("| " + " | ".join(full_row) + " |")
        
        # 如果数据行超过31行，只显示前31行
        if len(rows) > 31:
            result_lines.append("| ... | ... |")
            result_lines.append(f"| 共 {len(rows)} 行数据，仅显示前31行 |")
        
        return '\n'.join(result_lines)
    
    def extract_numbers(self, text):
        """提取所有数字"""
        numbers = re.findall(r'\d+\.?\d*', text)
        return ' '.join(numbers)
    
    def update_result(self, text):
        """更新结果显示"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self.update_word_count()
        
        if text.strip():
            self.add_to_history(text)
    
    def update_word_count(self, event=None):
        """更新统计信息"""
        text = self.result_text.get(1.0, tk.END).strip()
        lines = text.count('\n') + 1 if text else 0
        self.word_count_label.config(text=f"数据行: {lines}")
    
    def copy_result(self):
        text = self.result_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_bar.config(text="已复制到剪贴板")
            messagebox.showinfo("成功", "结果已复制到剪贴板")
        else:
            messagebox.showwarning("警告", "没有内容可复制")
    
    def save_result(self):
        if not self.result_text.get(1.0, tk.END).strip():
            messagebox.showwarning("警告", "没有结果可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存识别结果",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"===== OCR 表格识别结果 =====\n")
                f.write(f"识别时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"图片文件: {os.path.basename(self.current_image_path) if self.current_image_path else '未知'}\n")
                f.write(f"语言: {self.lang_var.get()}\n")
                f.write(f"模式: {self.mode_var.get()}\n")
                f.write(f"预处理: {self.preprocess_var.get()}\n")
                f.write("=" * 40 + "\n\n")
                f.write(self.result_text.get(1.0, tk.END))
            
            # 如果是CSV格式
            if file_path.endswith('.csv'):
                self.save_as_csv(file_path)
            
            messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")
            self.status_bar.config(text=f"已保存: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def save_as_csv(self, file_path):
        """保存为CSV格式"""
        try:
            text = self.result_text.get(1.0, tk.END).strip()
            lines = text.split('\n')
            csv_lines = []
            
            for line in lines:
                if line.startswith('|') and line.endswith('|'):
                    # 解析表格行
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if cells and not all(c == '------' for c in cells):
                        csv_lines.append(','.join(cells))
            
            if csv_lines:
                csv_path = file_path.replace('.txt', '.csv')
                with open(csv_path, 'w', encoding='utf-8-sig') as f:
                    f.write('\n'.join(csv_lines))
        except:
            pass
    
    def clear_all(self):
        self.current_image = None
        self.current_image_path = None
        self.zoom_level = 1.0
        self.canvas.delete("all")
        self.image_on_canvas = None
        self.result_text.delete(1.0, tk.END)
        self.word_count_label.config(text="数据行: 0")
        self.image_info.config(text="图片信息: 未加载")
        self.status_bar.config(text="已清空")
    
    def add_to_history(self, text):
        if not hasattr(self, 'history'):
            self.history = []
        
        entry = {
            'time': datetime.datetime.now().isoformat(),
            'text': text[:200] + '...' if len(text) > 200 else text,
            'full_text': text
        }
        self.history.insert(0, entry)
        
        if len(self.history) > 100:
            self.history = self.history[:100]
        
        self.save_history()
    
    def load_history(self):
        history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.json')
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def batch_process(self, folder_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(image_extensions)]
        
        if not images:
            messagebox.showwarning("警告", "文件夹中没有图片")
            return
        
        output_dir = os.path.join(folder_path, 'ocr_results')
        os.makedirs(output_dir, exist_ok=True)
        
        self.progress_var.set(0)
        self.status_bar.config(text=f"正在处理 0/{len(images)} 个图片...")
        
        def process():
            total = len(images)
            for idx, img_file in enumerate(images, 1):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = Image.open(img_path)
                    processed_img = self.apply_preprocessing(img)
                    
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(processed_img, 
                                                      lang="eng",
                                                      config=custom_config)
                    
                    if self.mode_var.get() == "表格数字":
                        text = self.process_table_data(text)
                    
                    txt_file = os.path.join(output_dir, 
                                          f"{os.path.splitext(img_file)[0]}.txt")
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    progress = (idx / total) * 100
                    self.root.after(0, lambda p=progress, i=idx, t=img_file: 
                        (self.progress_var.set(p),
                         self.status_bar.config(text=f"进度: {i}/{total} - {t}")))
                    
                except Exception as e:
                    print(f"处理 {img_file} 失败: {e}")
            
            self.root.after(0, lambda: messagebox.showinfo("完成", 
                f"批量处理完成！\n共处理 {total} 个图片\n结果保存在: {output_dir}"))
            self.root.after(0, lambda: self.status_bar.config(text="批量处理完成"))
            self.root.after(0, lambda: self.progress_var.set(0))
        
        threading.Thread(target=process, daemon=True).start()

def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except:
        root = tk.Tk()
    
    app = OCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()