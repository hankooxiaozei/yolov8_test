import glob
import os

import cv2
import numpy as np

# --- 辅助函数 (这部分没有变化) ---


def denoise_with_bilateral_filter(image, d, sigma_color, sigma_space):
    """双边滤波去噪."""
    if d == 0:
        return image.copy()
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast_clahe(image, clipLimit, tileGridSize):
    """CLAHE对比度增强."""
    if clipLimit == 0 or tileGridSize[0] == 0:
        return image.copy()

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_l_channel = clahe.apply(l_channel)
    merged_lab_image = cv2.merge([clahe_l_channel, a_channel, b_channel])
    enhanced_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_image


def sharpen_image(image):
    """使用拉普拉斯算子进行简单锐化."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


# --- 核心预处理流程 (这部分没有变化) ---


def preprocess_pipeline(image, d, sigma_color, sigma_space, clip_limit, tile_size, use_sharpen):
    denoised_img = denoise_with_bilateral_filter(image, d, sigma_color, sigma_space)
    square_grid_size = (tile_size, tile_size)
    enhanced_img = enhance_contrast_clahe(denoised_img, clip_limit / 10.0, square_grid_size)
    if use_sharpen > 0:
        enhanced_img = sharpen_image(enhanced_img)
    return enhanced_img


# --- 交互式调参主函数 ---


def interactive_check_labels(txt_labels_dir, images_dir):
    txt_files = glob.glob(os.path.join(txt_labels_dir, "*.txt"))
    if not txt_files:
        print(f"在 {txt_labels_dir} 中未找到任何 .txt 标签文件。")
        return

    window_name = "Preprocessing Workbench"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 创建滑动条...
    cv2.createTrackbar("d", window_name, 9, 50, lambda x: None)
    cv2.createTrackbar("sigmaColor", window_name, 75, 200, lambda x: None)
    cv2.createTrackbar("sigmaSpace", window_name, 75, 200, lambda x: None)
    cv2.createTrackbar("clipLimit*10", window_name, 20, 100, lambda x: None)
    cv2.createTrackbar("tileSize", window_name, 8, 32, lambda x: None)
    cv2.createTrackbar("Sharpen (0/1)", window_name, 0, 1, lambda x: None)

    index = 0
    pic_path = ""

    while True:
        # (文件加载部分保持不变)
        txt_file = txt_files[index]
        filename = os.path.splitext(os.path.basename(txt_file))[0]
        possible_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        original_img = None
        found_image = False
        for ext in possible_exts:
            current_pic_path = os.path.join(images_dir, filename + ext)
            if os.path.exists(current_pic_path):
                original_img = cv2.imread(current_pic_path)
                pic_path = current_pic_path
                found_image = True
                break
        if not found_image:
            print(f"警告：无法为标签 {filename} 找到对应的图像文件。")
            index = (index + 1) % len(txt_files)
            continue

        # (获取滑动条值和应用预处理流程部分保持不变)
        d = cv2.getTrackbarPos("d", window_name)
        sigma_color = cv2.getTrackbarPos("sigmaColor", window_name)
        sigma_space = cv2.getTrackbarPos("sigmaSpace", window_name)
        clip_limit = cv2.getTrackbarPos("clipLimit*10", window_name)
        tile_size = cv2.getTrackbarPos("tileSize", window_name)
        use_sharpen = cv2.getTrackbarPos("Sharpen (0/1)", window_name)
        if tile_size == 0:
            tile_size = 1
        processed_img = preprocess_pipeline(
            original_img.copy(), d, sigma_color, sigma_space, clip_limit, tile_size, use_sharpen
        )

        # (绘制标签部分保持不变)
        height, width, _ = processed_img.shape
        try:
            with open(txt_file) as file_handle:
                cnt_info = file_handle.readlines()
            new_cnt_info = [line.strip().split(" ") for line in cnt_info]
            color_map = {"0": (0, 255, 255), "1": (255, 0, 255)}
            for new_info in new_cnt_info:
                class_id = new_info[0]
                points = []
                for i in range(1, len(new_info), 2):
                    x_norm, y_norm = float(new_info[i]), float(new_info[i + 1])
                    points.append([int(x_norm * width), int(y_norm * height)])
                cv2.polylines(original_img, [np.array(points, np.int32)], True, color_map.get(class_id, (0, 0, 255)), 2)
                cv2.polylines(
                    processed_img, [np.array(points, np.int32)], True, color_map.get(class_id, (0, 0, 255)), 2
                )
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")

        # --- 显示结果 ---
        combined_display = np.hstack((original_img, processed_img))

        display_filename = os.path.basename(pic_path)
        info_text = (
            f"File: {display_filename} | d={d}, sC={sigma_color}, sS={sigma_space}, "
            f"clip={clip_limit / 10.0:.1f}, tile=({tile_size}x{tile_size}), sharpen={use_sharpen}"
        )

        # ==================== 主要修改处：调整字体参数 ====================
        # 你可以在这里自由调整这些值，直到满意为止
        font_scale = 4.0  # 字体大小（之前是0.8）
        text_thickness = 5  # 文字粗细（之前是2）
        outline_thickness = 4  # 描边粗细（之前是3）
        text_position = (80, 100)  # 文字位置(x, y)，向下移动了一些
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)  # 白色
        outline_color = (0, 0, 0)  # 黑色

        # 使用定义的变量来绘制带描边的文本
        cv2.putText(
            combined_display,
            info_text,
            text_position,
            font_face,
            font_scale,
            outline_color,
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined_display, info_text, text_position, font_face, font_scale, font_color, text_thickness, cv2.LINE_AA
        )
        # ================================================================

        cv2.imshow(window_name, combined_display)

        # (按键控制部分保持不变)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            index = (index + 1) % len(txt_files)
        elif key == ord("a"):
            index = (index - 1 + len(txt_files)) % len(txt_files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    save_dir = "C:/Users/HL/Downloads/wendang_labels/label_test/"
    img_dir = "C:/Users/HL/Downloads/wendang_labels/images20250826/"
    interactive_check_labels(save_dir, img_dir)
