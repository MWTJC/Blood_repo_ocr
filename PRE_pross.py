# coding=utf-8
import os
import math
import re

import numpy as np
import numpy
import cv2
# import timeout_decorator
import configparser
import matplotlib.pyplot as plt
from PIL import Image

from colorama import init, Fore, Back
init(autoreset=True)


class configparser_custom(configparser.ConfigParser):  # 解决默认被转换为小写问题
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr


# 透视变换类
def order_points(pts):
    """

    """
    # 一共四个坐标点
    rect = np.zeros((4, 2), dtype='float32')
    # 按顺序找到对应的坐标0123 分别是左上，右上，右下，左下
    # 计算左上，由下
    # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值
    s = pts.sum(axis=1)  # [2815.2   1224.    2555.712 3902.112]
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左
    # np.diff()  沿着指定轴计算第N维的离散差值  后者-前者
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # print('校正坐标：', rect)
    return rect


def four_point_transform(image, pts):
    """
    此处导入需要进行透视变换的图片与具体变换坐标
    返回cv2格式的变换后的图片
    :return: cv2.img
    """
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h的值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变化后对应坐标位置，需求案例，异常处理逻辑，通信协议，多任务（通信），实验报告（测试报告，通信协议）视频， 工程代码
    # 两个文档，工程代码包，16周5下午截止
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype='float32')

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


# 透视变换结束


# 核心
def knn_match_new(template_img, img_need_match, demo):
    MIN_MATCH_COUNT = 10
    # SIFT检测角点
    sift = cv2.SIFT_create()
    # 关键点和特征值
    kp1, des1 = sift.detectAndCompute(template_img, None)
    kp2, des2 = sift.detectAndCompute(img_need_match, None)
    # FLANN匹配
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 使用KNN算法匹配
    matches = flann.knnMatch(des1, des2, k=2)
    # 去除错误匹配
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 单应性
    if len(good) > MIN_MATCH_COUNT:
        result = 1
        print(Fore.LIGHTBLUE_EX+f"匹配结果 - {len(good)}/{MIN_MATCH_COUNT}")
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            print(Back.RED+'畸形匹配')
            dst = 0
            result = 2874734
            return dst, result

        if demo == 1:
            # ravel方法将数据降维处理，最后并转换成列表格式
            matchesMask = mask.ravel().tolist()
            # 获取img1的图像尺寸
            h, w = template_img.shape
            # pts是图像img1的四个顶点
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # 计算变换后的四个顶点坐标位置
            dst = cv2.perspectiveTransform(pts, M)
            # print(dst)
            # 画出变换后的边框
            img_need_match = cv2.polylines(img_need_match, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)

    else:
        print(Fore.RED+f"不甚匹配 - {len(good)}/{MIN_MATCH_COUNT}")
        if demo == 1:
            plt.imshow(img_need_match, 'gray'), plt.show()
        dst = 0
        result = 2874734
        return dst, result
        # matchesMask = None
    if demo == 1:
        # 显示匹配结果
        draw_params = dict(matchColor=(0, 255, 0),  # 绿色绘制线条
                           singlePointColor=None,
                           matchesMask=matchesMask,  # 仅绘制有效匹配
                           flags=2)
        img3 = cv2.drawMatches(template_img, kp1, img_need_match, kp2, good, None, **draw_params)
        cv2.imwrite('temp/DEMO/knn.jpg', img3)

    # print('匹配完毕...')
    return np.linalg.inv(M), result


def get_knn_result(orig, screenCnt, flip_or_not, demo_or_not):
    # 透视变换  坐标缩放
    warped = four_point_transform(orig, screenCnt.reshape(4, 2))
    if flip_or_not == 1:  # 配置文件指定反色
        warped = flip(warped)
    '''  
    if demo_or_not == 1:
        cv2.imwrite('TEMP/scan_result.jpg', warped)  # 一级输出
    '''
    return warped
    # print(ratio)


# 长宽比校正
def length_width_ratio_correct(img_template, img_input):
    x_template, y_template = img_template.shape[:2]
    x_input, y_input = img_input.shape[:2]
    shape_template = x_template / y_template
    img_output = cv2.resize(img_input, (int(x_input / shape_template), int(x_input)), interpolation=cv2.INTER_CUBIC)
    return img_output


# 图片颜色类
def flip(src):  # 反色
    fip = cv2.bitwise_not(src)
    return fip


def erosion(img):  # 腐蚀
    kernel = np.ones((2, 2), np.uint8)
    ero = cv2.erode(img, kernel, iterations=1)
    return ero


def dilate(img):  # 膨胀
    kernel = np.ones((2, 2), np.uint8)
    dil = cv2.dilate(img, kernel, iterations=1)
    return dil


def gamma(cvimg):
    def gamma_core(img, gamma_val):  # gamma函数处理
        gamma_table = [np.power(x / 255.0, gamma_val) * 255.0 for x in range(256)]  # 建立映射表
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
        return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

    # img = cv2.imread(file_path)  # 原图读取
    img_gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma_val = math.log10(0.9) / math.log10(mean / 255)  # 公式计算gamma
    # 0到1，默认0.5 纸面处理建议0.9，越大越亮
    image_gamma_correct = gamma_core(cvimg, gamma_val)  # gamma变换
    # print(mean, np.mean(image_gamma_correct))
    return image_gamma_correct


# 图片基本变换
def zoom_to_2k(img):
    height, width = img.shape[:2]
    if (height < 1500) or (height < 1500):
        print('试图放大到最少1.5k...')
        if height < width:
            ratio = 1500 / height
            size = (int(width * ratio), int(height * ratio))
        else:
            ratio = 1500 / width
            size = (int(width * ratio), int(height * ratio))
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    else:
        img = img
        ratio = 1
        print('未放大...')
    return img, ratio


def zoom_to_1k(img):
    height, width = img.shape[:2]
    '''
    if (height < 1000) or (height < 1000):
        # print('**-WARN:屏幕区域太小-**')
        '''
    if height > width:
        ratio = 1000 / height
        size = (int(width * ratio), int(height * ratio))
    else:
        ratio = 1000 / width
        size = (int(width * ratio), int(height * ratio))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    '''
    else:
        img = img
        ratio = 1
        print('未缩小...')
    '''
    return img, ratio


def crop_xls_zoom_new(boxes_coordinate_xy, scaling_ratio):  # 坐标读取并缩放
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for k in range(len(boxes_coordinate_xy)):
        x0.append(boxes_coordinate_xy[k][0])
        x1.append(boxes_coordinate_xy[k][1])
        y0.append(boxes_coordinate_xy[k][2])
        y1.append(boxes_coordinate_xy[k][3])
        if x0[k] > x1[k]:
            x0[k], x1[k] = x1[k], x0[k]
        if y0[k] > y1[k]:
            y0[k], y1[k] = y1[k], y0[k]
    y0a = [round(x / scaling_ratio) for x in y0]
    y1a = [round(x / scaling_ratio) for x in y1]
    x0a = [round(x / scaling_ratio) for x in x0]
    x1a = [round(x / scaling_ratio) for x in x1]
    return y0a, y1a, x0a, x1a


def mask_processing_new(img_input, boxes_coordinate_xy, demo_or_not, type_char, out_name):
    if demo_or_not == 1:
        print('剪裁读入：', img_input.shape[:2])
    image_1k, ratio = zoom_to_1k(img_input)  # xls坐标以1k为标准，将坐标缩放，适配图片，此处确定缩放比例
    y0a, y1a, x0a, x1a = crop_xls_zoom_new(boxes_coordinate_xy=boxes_coordinate_xy,
                                           scaling_ratio=ratio)
    dst = []
    for k in range(len(boxes_coordinate_xy)):
        dst.append(img_input[int(y0a[k]):int(y1a[k]), int(x0a[k]):int(x1a[k])])  # 裁剪

        if demo_or_not == 1:
            n = f'temp/DEMO/mask/' + out_name + f'_{type_char}_num{k:01}.jpg'
            cv2.imwrite(n, dst[k])  # 二级输出
    return dst


def image_border(img_input, dst):
    '''
    src: (str) 需要加边框的图片路径
    dst: (str) 加边框的图片保存路径
    loc: (str) 边框添加的位置, 默认是'a'(
        四周: 'a' or 'all'
        上: 't' or 'top'
        右: 'r' or 'rigth'
        下: 'b' or 'bottom'
        左: 'l' or 'left'
    )
    width: (int) 边框宽度 (默认是3)
    color: (int or 3-tuple) 边框颜色 (默认是0, 表示黑色; 也可以设置为三元组表示RGB颜色)
    '''
    # 拓宽倍率（按照最长边计算）
    muti = 1.10
    # cv2转PIL
    img_ori = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))

    color = (255, 255, 255)  # 定义白色底板
    # 读取图片
    # img_ori = Image.open(src)
    w = img_ori.size[0]
    h = img_ori.size[1]

    # 判断边框
    diff = w - h
    # width = int(abs(diff / 2))
    if diff >= 0:  # 如果宽大于高
        square_blank_width = int(float(w)*muti)
        add_h = int((square_blank_width-h)*0.5)
        add_w = int((square_blank_width - w) * 0.5)
        # 加top
        h += add_h
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, add_h, w, h))
        img_ori = img_new
        # 加botton
        h += add_h
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h - add_h))
        # 再加一点
        img_ori = img_new
        # 加left
        w += add_w
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (add_w, 0, w, h))
        img_ori = img_new
        # 加right
        w += add_w
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w - add_w, h))
    elif diff < 0:
        square_blank_width = int(float(h) * muti)
        add_h = int((square_blank_width - h) * 0.5)
        add_w = int((square_blank_width - w) * 0.5)
        # 加left
        w += add_w
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (add_w, 0, w, h))
        img_ori = img_new
        # 加right
        w += add_w
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w - add_w, h))
        # 再加一点
        img_ori = img_new
        # 加top
        h += add_h
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, add_h, w, h))
        img_ori = img_new
        # 加botton
        h += add_h
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h - add_h))
    else:
        pass

    '''
    # 添加边框
    if loc in ['a', 'all']:
        w += 2*width
        h += 2*width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, width))
    elif loc in ['t', 'top']:
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, width, w, h))
    elif loc in ['r', 'right']:
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w-width, h))
    elif loc in ['b', 'bottom']:
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h-width))
    elif loc in ['l', 'left']:
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, 0, w, h))
    else:
        pass
    '''
    # 保存图片
    # img_new.save(dst)
    # PIL转CV2
    img_ret = cv2.cvtColor(numpy.asarray(img_new), cv2.COLOR_RGB2BGR)
    return img_ret


def charactor_match_hospital_name(result_list, charactor_need_match):
    regex_str = f".*?([\u4E00-\u9FA5]+{charactor_need_match})"
    for i in range(len(result_list)):
        match_obj = re.match(regex_str, result_list[i])
        if match_obj:
            print(Fore.GREEN+'命中:'+match_obj.group(1))
            break
    else:
        print(Fore.YELLOW+f'未命中:{charactor_need_match}')
        return None
    return match_obj.group(1)


def charactor_match_count_name_age(result_list, charactor_need_match):
    regex_str = f"({charactor_need_match}.*).*"
    for i in range(len(result_list)):
        match_obj = re.search(regex_str, result_list[i])
        if match_obj:
            print(Fore.GREEN+'命中:'+match_obj.group(1))
            break
    else:
        print(Fore.YELLOW+f'未命中:{charactor_need_match}')
        return None
    return match_obj.group(1)


def charactor_match_any(result_list, charactor_need_match):
    regex_str = f".*{charactor_need_match}.*"
    for i in range(len(result_list)):
        match_obj = re.search(regex_str, result_list[i])
        if match_obj:
            print(Fore.GREEN+'命中:'+charactor_need_match+'->'+match_obj.string)
            return match_obj.string
    else:
        print(Fore.YELLOW+f'未命中:{charactor_need_match}')
        return None
    # return match_obj.group(1)


def charactor_match_chinese_head(result_list):
    regex_str = '^[\u4E00-\u9FA5].*'
    match_obj = re.match(regex_str, result_list)
    if match_obj:
        return True
    else:
        return False


def charactor_match_count_sex(result_list, charactor_need_match):
    regex_str = f"({charactor_need_match}.*).*"
    for i in range(len(result_list)):
        match_obj = re.search(regex_str, result_list[i])
        if match_obj:
            match_obj_black = re.search('.*自费', result_list[i])
            if match_obj_black:
                continue
            print(Fore.GREEN+'命中:'+match_obj.group(1))
            break
    else:
        print(Fore.YELLOW+f'未命中:{charactor_need_match}')
        return None
    return match_obj.group(1)


def cv_imread_chs(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

'''
def where_is_work_folder():
    path = os.path.dirname(os.path.abspath(__file__))  # C:\\project\\dist\\WEB_MAIN'
    root_path = path+'\\'
    return root_path
'''
