import glob
import os

import cv2
import numpy as np


def check_labels(txt_labels, images_dir):
    txt_files = glob.glob(txt_labels + "/*.txt")
    for txt_file in txt_files:
        filename = os.path.splitext(os.path.basename(txt_file))[0]

        # pic_path = images_dir + filename + ".jpg"
        pic_path = os.path.join(images_dir, filename + ".jpg")  # 使用 os.path.join 构建完整图像路径
        img = cv2.imread(pic_path)
        img = denoise_with_bilateral_filter(img)
        img = enhance_contrast_clahe(img, 2.5, (8, 8))
        # img = enhance_with_histogram_equalization(img)
        # img = sharpen_image(img)

        height, width, _ = img.shape

        file_handle = open(txt_file)
        cnt_info = file_handle.readlines()
        new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]

        color_map = {"0": (0, 255, 255)}
        for new_info in new_cnt_info:
            print(new_info)
            s = []
            for i in range(1, len(new_info), 2):
                b = [float(tmp) for tmp in new_info[i : i + 2]]
                s.append([int(b[0] * width), int(b[1] * height)])
            cv2.polylines(img, [np.array(s, np.int32)], True, color_map.get(new_info[0]))
        cv2.namedWindow("img2", 0)
        cv2.imshow("img2", img)
        cv2.waitKey()


# =============================================================================
# 步骤 1: 使用双边滤波去除高斯噪声
# =============================================================================
def denoise_with_bilateral_filter(image):
    """
    使用双边滤波对图像进行去噪处理。 参数:

    image (numpy.ndarray): 输入的含噪图像 (BGR格式)。 返回: numpy.ndarray: 去噪后的图像。.
    """
    print("步骤 1: 正在使用双边滤波进行去噪...")

    # cv2.bilateralFilter() 函数有4个主要参数:
    # 1. src: 输入图像
    # 2. d: 滤波时每个像素邻域的直径。值为负数时，由 sigmaSpace 自动计算。一般设为5-9。
    # 3. sigmaColor: 色彩空间滤波器的sigma值。该值越大，表示邻域内有更大色彩差异的像素会被归一化，从而产生更大范围的半相等颜色。
    # 4. sigmaSpace: 坐标空间滤波器的sigma值。该值越大，意味着越远的像素会相互影响，只要它们的颜色足够接近。
    #
    # 简单来说:
    # - sigmaSpace 控制空间距离权重，决定了滤波的“范围”。
    # - sigmaColor 控制色彩/灰度距离权重，决定了是否“保护边缘”。

    # 这些参数可以根据实际噪声水平进行调整
    d = 5
    sigma_color = 50
    sigma_space = 50
    denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return denoised_image


# =============================================================================
# 步骤 2: 使用分通道直方图均衡化进行图像增强
# =============================================================================
def enhance_with_histogram_equalization(image):
    """
    通过对RGB三通道分别进行直方图均衡化来增强图像对比度。 参数:

    image (numpy.ndarray): 输入图像 (BGR格式)。 返回: numpy.ndarray: 增强后的图像。.
    """
    print("步骤 2: 正在使用分通道直方图均衡化进行增强...")
    # 1. 将BGR图像分解为B, G, R三个独立的通道
    # 注意：OpenCV默认的颜色顺序是 BGR
    b, g, r = cv2.split(image)
    # 2. 对每个通道分别应用直方图均衡化
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # 3. 将处理后的三个通道合并回一个BGR图像
    enhanced_image = cv2.merge([b_eq, g_eq, r_eq])
    return enhanced_image


def enhance_contrast_clahe(image, clipLimit=4.0, tileGridSize=(8, 8)):
    """
    使用CLAHE在LAB颜色空间上增强图像对比度。 参数:

    image (numpy.ndarray): 输入的BGR格式图像。 返回: numpy.ndarray: 对比度增强后的BGR格式图像。.
    """
    # 1. 将图像从BGR颜色空间转换到LAB颜色空间
    # LAB空间将亮度和颜色信息分开
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # 2. 将LAB图像分割成L, A, B三个通道
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # 3. 创建CLAHE对象并设置参数
    # clipLimit: 颜色对比度限制，防止过度放大噪声。值越大对比度越强。
    # tileGridSize: 定义了进行直方图均衡化的网格大小。
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    # 4. 对L（亮度）通道应用CLAHE
    # 我们只增强亮度，不改变颜色信息
    clahe_l_channel = clahe.apply(l_channel)
    # 5. 将增强后的L通道与原始的A和B通道合并
    merged_lab_image = cv2.merge([clahe_l_channel, a_channel, b_channel])
    # 6. 将合并后的LAB图像转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_image


# 锐化函数
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='json convert to txt params')
    # parser.add_argument('--json-dir', type=str, default='dataset/json_labels', help='json path dir')
    # parser.add_argument('--save-dir', type=str, default='dataset/labels', help='txt save dir')
    # parser.add_argument('--classes', type=str, default='surface', help='classes')
    # args = parser.parse_args()
    # json_dir = args.json_dir
    # save_dir = args.save_dir
    # classes = args.classes

    save_dir = "C:/Users/HL/Downloads/wendang_labels/label_test/"
    img_dir = "C:/Users/HL/Downloads/wendang_labels/images20250826/"
    check_labels(save_dir, img_dir)
