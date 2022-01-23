import math
import re

import numpy as np
import numpy
import cv2
# import timeout_decorator
import configparser
import matplotlib.pyplot as plt
from PIL import Image


class configparser_custom(configparser.ConfigParser):  # 解决默认被转换为小写问题
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr


# 透视变换类
def order_points(pts):
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

    # 变化后对应坐标位置
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
def knn_match_new(img_des, img_need_knn, demo):
    MIN_MATCH_COUNT = 10
    # SIFT检测角点
    sift = cv2.SIFT_create()
    # 关键点和特征值
    kp1, des1 = sift.detectAndCompute(img_des, None)
    kp2, des2 = sift.detectAndCompute(img_need_knn, None)
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
        print("匹配结果 - %d/%d" % (len(good), MIN_MATCH_COUNT))
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            print('畸形匹配')
            dst = 0
            result = 2874734
            return dst, result

        if demo == 1:
            # ravel方法将数据降维处理，最后并转换成列表格式
            matchesMask = mask.ravel().tolist()
            # 获取img1的图像尺寸
            h, w = img_des.shape
            # pts是图像img1的四个顶点
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # 计算变换后的四个顶点坐标位置
            dst = cv2.perspectiveTransform(pts, M)
            # print(dst)
            # 画出变换后的边框
            img_need_knn = cv2.polylines(img_need_knn, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

    else:
        print("不甚匹配 - %d/%d" % (len(good), MIN_MATCH_COUNT))
        if demo == 1:
            plt.imshow(img_need_knn, 'gray'), plt.show()
        dst = 0
        result = 2874734
        return dst, result
        # matchesMask = None
    if demo == 1:
        # 显示匹配结果
        draw_params = dict(matchColor=(255, 255, 0),  # 黄线绘制变换框
                           singlePointColor=None,
                           matchesMask=matchesMask,  # 仅绘制有效
                           flags=2)
        img3 = cv2.drawMatches(img_des, kp1, img_need_knn, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()

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


'''
def ocr_recognition(img):

    text = pytesseract.image_to_string(img, lang='eng',
                                       config=' --psm 6 --oem 3 -c tessedit_char_whitelist=0123456789()/.ml%:-').strip()
    # print(text, file=data)
    return text
'''


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


def gamma_core(img, gamma_val):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma_val) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def gamma(img):
    # img = cv2.imread(file_path)  # 原图读取
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma_val = math.log10(0.9) / math.log10(mean / 255)  # 公式计算gamma
    # 默认0.5 建议0.2或者0.1，越大越亮
    image_gamma_correct = gamma_core(img, gamma_val)  # gamma变换
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


def crop_xls_zoom_new(boxes_xy, scaling_ratio):  # 坐标读取并缩放
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for k in range(len(boxes_xy)):
        x0.append(boxes_xy[k][0])
        x1.append(boxes_xy[k][1])
        y0.append(boxes_xy[k][2])
        y1.append(boxes_xy[k][3])
        if x0[k] > x1[k]:
            x0[k], x1[k] = x1[k], x0[k]
        if y0[k] > y1[k]:
            y0[k], y1[k] = y1[k], y0[k]
    y0a = [round(x / scaling_ratio) for x in y0]
    y1a = [round(x / scaling_ratio) for x in y1]
    x0a = [round(x / scaling_ratio) for x in x0]
    x1a = [round(x / scaling_ratio) for x in x1]
    return y0a, y1a, x0a, x1a


def mask_processing_new(image, boxes_xy, demo_or_not, type_char, output_dir, out_name):
    if demo_or_not == 1:
        print('剪裁读入：', image.shape[:2])
    image_1k, ratio = zoom_to_1k(image)  # xls坐标以1k为标准，将坐标缩放，适配图片，此处确定缩放比例
    y0a, y1a, x0a, x1a = crop_xls_zoom_new(boxes_xy, ratio)
    dst = []
    for k in range(len(boxes_xy)):
        dst.append(image[int(y0a[k]):int(y1a[k]), int(x0a[k]):int(x1a[k])])  # 裁剪

        if demo_or_not == 1:
            n = output_dir + f'mask/' + out_name + f'_{type_char}_num{k:01}.jpg'
            cv2.imwrite(n, dst[k])  # 二级输出
    return dst


def image_border(src, dst):
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
    # cv2转PIL
    img_ori = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    color = (255, 255, 255)
    # 读取图片
    # img_ori = Image.open(src)
    w = img_ori.size[0]
    h = img_ori.size[1]

    # 判断边框
    diff = w - h
    width = int(abs(diff/2))
    if diff >= 0:
        # 加top
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, width, w, h))
        img_ori = img_new
        # 加botton
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h - width))
        # 再加一点
        img_ori = img_new
        # 加left
        w += 20
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (20, 0, w, h))
        img_ori = img_new
        # 加right
        w += 20
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w - 20, h))
    elif diff < 0:
        # 加left
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, 0, w, h))
        img_ori =img_new
        # 加right
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w - width, h))
        # 再加一点
        img_ori = img_new
        # 加top
        h += 20
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 20, w, h))
        img_ori = img_new
        # 加botton
        h += 20
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0, w, h - 20))
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
    #img_new.save(dst)
    # PIL转CV2
    img_ret = cv2.cvtColor(numpy.asarray(img_new), cv2.COLOR_RGB2BGR)
    return img_ret


def charactor_match_hospital_name(result_list, charactor_need_match):
    regex_str = f".*?([\u4E00-\u9FA5]+{charactor_need_match})"
    for i in range(len(result_list)):
        match_obj = re.match(regex_str, result_list[i])
        if match_obj:
            print(match_obj.group(1))
            break
    else:
        print(f'未能判断:{charactor_need_match}')
        return None
    return match_obj.group(1)


'''
def match():
    pattern = re.compile(ur'.+([\u4E00-\u9FA5]+医院).+')
    str = u''
    print(pattern.search(str))
'''


def charactor_match_count_name_age(result_list, charactor_need_match):
    regex_str = f"({charactor_need_match}.*).*"
    for i in range(len(result_list)):
        match_obj = re.search(regex_str, result_list[i])
        if match_obj:
            print(match_obj.group(1))
            break
    else:
        print(f'未能判断:{charactor_need_match}')
        return None
    return match_obj.group(1)


def cv_imread_chs(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
