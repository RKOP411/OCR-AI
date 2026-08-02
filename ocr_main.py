# ocr_main.py
import os
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import pytesseract
import threading
import datetime

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR 文字识别工具")
        self.root.geometry("800x600")
        
        # 设置Tesseract路径 - 使用相对路径
        self.setup_tesseract()
        
        # 创建UI
        self.create_widgets()
        
        # 当前打开的图片
        self.current_image = None
        self.current_image_path = None
    
    def setup_tesseract(self):
        """配置Tesseract路径，优先使用便携版"""
        # 获取程序所在目录
        if getattr(sys, 'frozen', False):
            # 如果是打包后的exe
            base_path = sys._MEIPASS
        else:
            # 如果是Python脚本
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 便携版Tesseract路径
        tesseract_path = os.path.join(base_path, 'Tesseract', 'tesseract.exe')
        
        # 检查便携版是否存在
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✅ 使用便携版Tesseract: {tesseract_path}")
        else:
            # 如果便携版不存在，尝试使用系统安装的版本
            try:
                pytesseract.get_tesseract_version()
                print("✅ 使用系统安装的Tesseract")
            except:
                messagebox.showerror("错误", 
                    "找不到Tesseract！\n"
                    "请确保Tesseract文件夹在程序目录下，\n"
                    "或者安装Tesseract到系统。")
                sys.exit(1)
    
    def create_widgets(self):
        """创建UI组件"""
        # 工具栏
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 按钮
        tk.Button(toolbar, text="📁 选择图片", command=self.select_image).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="📂 选择文件夹", command=self.select_folder).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="🔍 识别", command=self.start_ocr).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="💾 保存结果", command=self.save_result).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="🧹 清空", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        
        # 语言选择
        tk.Label(toolbar, text="语言:").pack(side=tk.LEFT, padx=(20, 5))
        self.lang_var = tk.StringVar(value="eng+chi_sim")
        lang_menu = tk.OptionMenu(toolbar, self.lang_var, 
                                  "eng", "chi_sim", "eng+chi_sim", "jpn", "kor")
        lang_menu.pack(side=tk.LEFT)
        
        # 主区域 - 左右分割
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左半部分 - 图片显示
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(left_frame, text="图片预览", font=('Arial', 10, 'bold')).pack()
        
        self.image_label = tk.Label(left_frame, text="请选择图片", 
                                    bg='#f0f0f0', relief=tk.RAISED)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右半部分 - 文字结果
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_frame, text="识别结果", font=('Arial', 10, 'bold')).pack()
        
        self.result_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """选择单个图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.load_image(file_path)
    
    def select_folder(self):
        """选择文件夹，批量处理"""
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        if folder_path:
            self.batch_process(folder_path)
    
    def load_image(self, image_path):
        """加载图片到预览区域"""
        try:
            self.current_image_path = image_path
            self.current_image = Image.open(image_path)
            
            # 显示缩略图
            img_copy = self.current_image.copy()
            # 调整大小以适应显示区域
            max_size = (400, 400)
            img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(img_copy)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            self.status_bar.config(text=f"已加载: {os.path.basename(image_path)}")
            
            # 自动识别
            self.start_ocr()
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
    def start_ocr(self):
        """开始OCR识别"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        # 在新线程中执行OCR，避免界面卡顿
        threading.Thread(target=self.do_ocr, daemon=True).start()
    
    def do_ocr(self):
        """执行OCR识别（在子线程中）"""
        try:
            self.status_bar.config(text="正在识别...")
            self.root.config(cursor="watch")
            
            # 获取语言
            lang = self.lang_var.get()
            
            # 执行OCR
            text = pytesseract.image_to_string(self.current_image, lang=lang)
            
            # 更新结果（在主线程中）
            self.root.after(0, lambda: self.update_result(text))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("OCR错误", str(e)))
        finally:
            self.root.after(0, lambda: self.status_bar.config(text="识别完成"))
            self.root.after(0, lambda: self.root.config(cursor=""))
    
    def update_result(self, text):
        """更新结果显示"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
    
    def save_result(self):
        """保存识别结果"""
        if not self.result_text.get(1.0, tk.END).strip():
            messagebox.showwarning("警告", "没有结果可保存")
            return
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.current_image_path:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            filename = f"{base_name}_{timestamp}.txt"
        else:
            filename = f"ocr_result_{timestamp}.txt"
        
        file_path = os.path.join(output_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.result_text.get(1.0, tk.END))
            messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")
            self.status_bar.config(text=f"已保存: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def clear_all(self):
        """清空所有"""
        self.current_image = None
        self.current_image_path = None
        self.image_label.config(image='', text="请选择图片")
        self.image_label.image = None
        self.result_text.delete(1.0, tk.END)
        self.status_bar.config(text="已清空")
    
    def batch_process(self, folder_path):
        """批量处理文件夹中的图片"""
        # 获取所有图片
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(image_extensions)]
        
        if not images:
            messagebox.showwarning("警告", "文件夹中没有图片")
            return
        
        # 创建输出目录
        output_dir = os.path.join(folder_path, 'ocr_results')
        os.makedirs(output_dir, exist_ok=True)
        
        self.status_bar.config(text=f"正在处理 {len(images)} 个图片...")
        
        # 在新线程中处理
        def process():
            for idx, img_file in enumerate(images, 1):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = Image.open(img_path)
                    text = pytesseract.image_to_string(img, lang=self.lang_var.get())
                    
                    # 保存结果
                    txt_file = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.txt")
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    self.root.after(0, lambda i=idx, t=img_file: 
                        self.status_bar.config(text=f"进度: {i}/{len(images)} - {t}"))
                    
                except Exception as e:
                    print(f"处理 {img_file} 失败: {e}")
            
            self.root.after(0, lambda: messagebox.showinfo("完成", 
                f"批量处理完成！\n共处理 {len(images)} 个图片\n结果保存在: {output_dir}"))
            self.root.after(0, lambda: self.status_bar.config(text="批量处理完成"))
        
        threading.Thread(target=process, daemon=True).start()

def main():
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()