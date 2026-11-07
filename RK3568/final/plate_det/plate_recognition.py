"""车牌识别与双层车牌拼接工具，兼容原有plate_recognition实现。"""

import cv2
import numpy as np


# 车牌字符字典
# 包含省份简称、字母、数字和特殊字符
# 共78类：42个省份/特殊字符、34个数字/字母以及2个扩展字符
# 不包含空格，字符数必须与模型输出维度严格一致
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"

# 识别模型预处理使用的归一化参数
mean_value = 0.588
std_value = 0.193

# 车牌颜色标签，顺序与模型输出索引一致
color_labels = ['黑色', '蓝色', '绿色', '白色', '黄色']


def get_split_merge(img):
    """把双层车牌拆分为上下两部分，统一高度后水平拼接供识别模型使用。

    参数：
        img: 已透视矫正的双层车牌BGR图像。

    返回：
        上下两层水平排列后的单行车牌图像。
    """
    h, w, c = img.shape
    mid = h * 5 // 12
    img_upper = img[0:mid, :]
    img_lower = img[mid:, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img
