def crop_cell_from_image(image_path, bbox):
    """根据外接矩形裁剪图像"""
    img = cv2.imread(image_path)
    x_min, y_min, x_max, y_max = bbox
    # 裁剪区域（注意坐标取整）
    crop = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    return crop