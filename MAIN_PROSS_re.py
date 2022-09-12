# coding=utf-8
from datetime import datetime
from string import digits
import re
import os
import shutil
from glob import glob
import numpy
import numpy as np
import requests
import cv2
import json
import base64
import PRE_pross
import configparser
from colorama import init, Fore, Back
init(autoreset=True)
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, OrderedDict


class configparser_custom(configparser.ConfigParser):  # 解决默认被转换为小写问题
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr

    def as_dict(self):
        """
        将configparser.ConfigParser().read()读到的数据转换成dict返回
        :return:
        """
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


class OCR_Pack(object):
    def __init__(self, knn_img_model: numpy.ndarray, knn_img_need_match: numpy.ndarray):
        self.img_model = knn_img_model
        self.img_need_match = knn_img_need_match


    def knn(img_model, img_need_match):
        MIN_MATCH_COUNT = 10
        # SIFT检测角点
        sift = cv2.SIFT_create()
        # 关键点和特征值
        kp1, des1 = sift.detectAndCompute(img_model, None)
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
            print(Fore.LIGHTBLUE_EX + f"匹配结果 - {len(good)}/{MIN_MATCH_COUNT}")
            # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # findHomography 函数是计算变换矩阵
            # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
            # 返回值：M 为变换矩阵，mask是掩模
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print(Back.RED + '畸形匹配')
                dst = 0
                result = 2874734
                return dst, result
            '''
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
            '''
        else:
            print(Fore.RED + f"不甚匹配 - {len(good)}/{MIN_MATCH_COUNT}")
            '''
            if demo == 1:
                plt.imshow(img_need_match, 'gray'), plt.show()
            '''
            dst = 0
            result = 2874734
            return dst, result
            # matchesMask = None
        '''
        if demo == 1:
            # 显示匹配结果
            draw_params = dict(matchColor=(0, 255, 0),  # 绿色绘制线条
                               singlePointColor=None,
                               matchesMask=matchesMask,  # 仅绘制有效匹配
                               flags=2)
            img3 = cv2.drawMatches(template_img, kp1, img_need_match, kp2, good, None, **draw_params)
            cv2.imwrite('temp/DEMO/knn.jpg', img3)
        '''
        # print('匹配完毕...')
        return np.linalg.inv(M), result


class Img(object):
    """
    此处导入使用class包装的CV格式图片和OCR识别服务器地址及端口，
    """
    # m_OCR_Pack = OCR_Pack()

    def __init__(self, img_f:numpy.ndarray, IP_Address: str):
        self.img_f = img_f
        self.IP_Address = IP_Address
        self.img_s = str

    def Send_To_OCR(self):  # 原net_ocr_class
        """
        此处导入使用class包装的CV格式图片和OCR识别服务器地址及端口，
        使用了py.requests包实现http发送功能，
        接收远程返回的json报文并直接返回。
        :return: str
        """

        def cv2_to_base64(image):
            """
            此处导入CV格式图片，
            使用了py.opencv.imdecode包实现图片转换，
            返回base64形式的图片信息。
            :return: str
            """
            data = cv2.imencode('.jpg', image)[1]
            return base64.b64encode(data.tobytes()).decode('utf8')

        # 发送HTTP请求
        data = {'images': [cv2_to_base64(self.img_f)]}
        headers = {"Content-type": "application/json"}
        # url = f"http://127.0.0.1:8866/predict/chinese_ocr_db_crnn_server"
        url = f"http://{self.IP_Address}/predict/chinese_ocr_db_crnn_mobile"
        try:
            img_s = requests.post(url=url, headers=headers, data=json.dumps(data))
        except ConnectionRefusedError:
            img_s = 'OCROFFLINE'
            print(img_s)
        except requests.exceptions.ConnectionError:
            img_s = 'OCROFFLINE'
            print(img_s)
        # pprint(img_s)
        return img_s


class Single_Check_Item(object):
    def __init__(self, Code_Name: str, Ratio: str, Refer_Range: str, Unit: str, Value: str):
        self.Code_Name = Code_Name
        self.Ratio = Ratio
        self.Refer_Range = Refer_Range
        self.Unit = Unit
        self.Value = Value


class Report_Paper(object):
    def __init__(self, Age: str, Check_Item: list, Hos_name: str, Name: str, Report_Date: str, Report_ID: int, Report_type: str, Sex: str):
        self.age = Age
        self.items = Check_Item
        self.hospital = Hos_name
        self.name = Name
        self.repo_date = Report_Date
        self.repo_ID = Report_ID
        self.repo_type = Report_type
        self.sex = Sex


class AnswerToReport(object):
    def __init__(self, ocr_answer: list, list_title: list):
        self.answer = ocr_answer
        self.list_title = list_title

    def to_list(self):

        def data_align_v2(input_body_list, list_title):
            """
            此处导入识别体list和每一列的数据名称
            :return: list
            """
            # todo 如何改进定义严格程度使更精准（单位像素）
            # 目前仅在合并无效行开头（项目名）使用了judge_new
            # 初始化输出列表
            list_out = []
            dict_out = []

            # 确定每行共有多少列
            number_of_columns = len(input_body_list)  # 此时input_list[numb...]和input_list[0]为检测项目名字，即对齐依据

            # number_of_columns = int(len(input_list)/2)  # 此时input_list[numb...]和input_list[0]为检测项目名字，即对齐依据

            def get_h_location_func(child_list_input):
                list_name_h_location_output = []
                for i in range(len(child_list_input)):  # 遍历第一次识别的所有结果

                    h_u = 0.5 * (float(child_list_input[i]['text_box_position'][0][1]) +
                                 float(child_list_input[i]['text_box_position'][1][1]))
                    h_d = 0.5 * (float(child_list_input[i]['text_box_position'][2][1]) +
                                 float(child_list_input[i]['text_box_position'][3][1]))
                    h_location = 0.5 * (h_u + h_d)  # 确定每一项的高度位置(取4个点的竖座标取平均值)
                    list_name_h_location_output.append(h_location)
                return list_name_h_location_output

            list_name_h_location = get_h_location_func(input_body_list[0][0]['data'])  # 确定项目名称高度坐标

            list_name_h_location_avg = []
            for i in range(len(list_name_h_location) - 1):
                list_name_h_location_avg.append(list_name_h_location[i + 1] - list_name_h_location[i])
            judge_new = sum(list_name_h_location_avg) / (len(list_name_h_location_avg))
            judge_new = judge_new * 0.3

            # 解决一行名字被识别为两项的情况（名字中间有过长空格，对处于同一高度的项目进行合并）
            list_invalid_name = []  # 记录无效项
            for i in range(len(list_name_h_location)):
                if i in list_invalid_name:
                    pass
                else:
                    for j in range(len(list_name_h_location)):
                        if j <= i:
                            pass
                        else:
                            if abs(list_name_h_location[i] - list_name_h_location[j]) < judge_new:
                                list_invalid_name.append(j)

            list_name_correct = []  # 经过确认的有效行头
            list_name_position_correct = []
            for i in range(len(input_body_list[0][0]['data'])):
                if i in list_invalid_name:
                    for n in range(4):  # 往前推，合并到最近的正常项
                        if i - 1 - n in list_invalid_name:
                            pass
                        else:
                            list_name_correct[-1] = list_name_correct[-1] + ' ' + (
                                input_body_list[0][0]['data'][i]['text'])  # 向前合并
                            break
                    pass
                else:
                    list_name_correct.append(input_body_list[0][0]['data'][i]['text'])  # 第一次组合，先填入行名字(检测项目名)
                    list_name_position_correct.append(list_name_h_location[i])
            # 无效检测项目合并完毕

            for i in range(len(list_name_correct)):  # 以下开始组合每一条数据
                list_out_child = []

                dict_out_child = dict()

                list_out_child.append(list_name_correct[i])  # 确定第一项：名字

                dict_out_child[list_title[0]] = list_name_correct[i]

                for k in range(1, number_of_columns):  # 按照竖行循环

                    if len(input_body_list[k][0]['data']) == 0:
                        list_out_child.append('空')

                        dict_out_child[list_title[k]] = '空'

                        pass
                    else:
                        list_diy_h_location = get_h_location_func(input_body_list[k][0]['data'])  # 确定其他项的高度

                        for l in range(len(list_diy_h_location)):  # 改为逐项匹配
                            column_head_diff = []
                            for m in range(len(list_name_position_correct)):
                                column_head_diff.append(abs(list_name_position_correct[m] - list_diy_h_location[l]))
                            # 以下开始寻找最近的项，对齐
                            for _ in range(2):
                                min_number = min(column_head_diff)
                                min_index = column_head_diff.index(min_number)
                                # column_head_diff[min_index] = 0  # 得到最小项
                            if i == min_index:
                                list_out_child.append(input_body_list[k][0]['data'][l]['text'])
                                dict_out_child[list_title[k]] = input_body_list[k][0]['data'][l]['text']

                        if len(list_out_child) == k:
                            list_out_child.append('空')

                            dict_out_child[list_title[k]] = '空'

                list_out.append(list_out_child)

                dict_out.append(dict_out_child)

            return list_out, dict_out

        # todo 改善对齐
        if int(len(self.answer)) % 2 == 1:
            print('需要对齐的列表有问题')
            return '错误:需要对齐的列表有问题'

        # todo 将坐标信息根据配置文件还原对齐
        '''
        # 因为使用了ocr图片加白边避免ocr不识别的bug，故此处仍需改进
        for i in range(len(answer)):
            for j in range(len(answer[i][0]['data'])):
                for k in range(len(answer[i][0]['data'][j]['text_box_position'])):
                    answer[i][0]['data'][j]['text_box_position'][k][0] = answer[i][0]['data'][j]['text_box_position'][k][0] + box_list[i][0]
                    answer[i][0]['data'][j]['text_box_position'][k][0] = answer[i][0]['data'][j]['text_box_position'][k][1] + box_list[i][2]
        '''

        answer1 = []
        answer2 = []
        list_title1 = []
        list_title2 = []
        for i in range(len(self.answer)):
            if i < int(len(self.answer) / 2):
                answer1.append(self.answer[i])
                list_title1.append(self.list_title[i])
            else:
                answer2.append(self.answer[i])
                list_title2.append(self.list_title[i])
        list_direct, dict_direct = data_align_v2(answer1, list_title1)
        list_direct2, dict_direct2 = data_align_v2(answer2, list_title2)
        list_direct.extend(list_direct2)
        dict_direct.extend(dict_direct2)

        bloodtest_list = []
        for i in range(len(dict_direct)):
            bloodtest_single = OrderedDict()
            for j in range(len(dict_direct[i])):
                try:
                    bloodtest_single[self.list_title[j]] = dict_direct[i][self.list_title[j]]
                except:
                    bloodtest_single[self.list_title[j]] = '空'
            bloodtest_list.append(bloodtest_single)
        return bloodtest_list


def read_keywords(path):
    """
    此处导入包含报告关键词的conf文件路径，
    返回读取到的关键词列表
    :return: list
    """
    keys = configparser_custom()
    keys.read(path, 'UTF-8')
    keys_read = keys.items("keywords")
    return keys_read


def type_judge(lstKwds_need_judge=list, conf_path=str):
    """
    此处导入包含报告关键词的conf文件路径，
    返回判断结果
    :return: str
    """
    path = f"{conf_path}/[*.conf"
    lstTxtFiles = glob(path)
    for strTxtFile in lstTxtFiles:
        keys_list = read_keywords(strTxtFile)
        keys_list_t = []
        for w in range(len(keys_list)):
            keys_list_t.append(keys_list[w][1])
        # strContent = txtWrapper.read()  # 读关键词
        i = 0.00
        n = 0.00
        for strKwd in keys_list_t:  # 用每个从本地读取到的关键词去匹配
            n = len(keys_list_t)
            if PRE_pross.charactor_match_any(lstKwds_need_judge, strKwd):
                # if strKwd in strContent:  # 如果命中
                i = i + 1
        print(i / n)
        if (i / n) > 0.4:
            # print(os.path.basename(strTxtFile))
            find = os.path.basename(strTxtFile)
            find_no_ex = find.split('.conf')
            find_type = find_no_ex[0].split(']')
            type = find_type[1]
            print(Back.GREEN + type)
            return type


def main_pross(cvimg, demo_or_not, hospital_lock, report_type_lock):

    img_org = cvimg
    img_gamma = PRE_pross.gamma(img_org)

    cv2.imwrite('temp/DEMO/gamma.jpg', img_gamma)

    class_report = Report_Paper  # 定义类结构

    OCR_IP_PATH = 'conf/OCR_IP.conf'
    conf_ocr_ip = configparser_custom()
    conf_ocr_ip.read(OCR_IP_PATH, 'UTF-8')
    ocr_ip_t = conf_ocr_ip.items("ip")
    ocr_ip = ocr_ip_t[0][1]

    # 判断所属医院以及检验项目
    img_gamma = PRE_pross.image_border(img_input=img_gamma,
                                       dst='0')
    '''
    ocr_pack = ClassSendtoOcr
    ocr_pack.img_send = img_gamma
    ocr_pack.address = ocr_ip
    pre_response = net_OCR_class(ocr_pack)
    '''

    img_org = Img(IP_Address=ocr_ip, img_f=cvimg)
    pre_response = img_org.Send_To_OCR()

    if pre_response is 'OCROFFLINE':
        return '错误：OCR离线'
    elif len(pre_response.json()["results"]) == 0:
        # print('OCRERR')
        return f'错误：OCR没有正常工作：\n{pre_response.json()["msg"]}'
    # pprint(pre_response.json()["results"][0]["data"])
    report_overview = []
    for i in range(len(pre_response.json()["results"][0]["data"])):
        report_overview.append(pre_response.json()["results"][0]["data"][i]['text'])

    # 取出医院关键词
    if hospital_lock == False:
        hospital = PRE_pross.charactor_match_hospital_name(report_overview, '医院')
        class_report.hos_name = PRE_pross.charactor_match_hospital_name(report_overview, '医院')  # class
    if hospital_lock == True:
        hospital = '复旦大学附属华山医院'
        class_report.hos_name = '复旦大学附属华山医院'  # class
    path_prefix = hospital
    # ocr会把间隔大的文字分开识别，大概率优先识别为中文字符

    if hospital is None:
        # print('未能识别医院信息')
        return '错误：未能识别所属医院，请拍摄完整的报告单图片，并保证纸面平整'

    # 读取报告类型关键词
    if report_type_lock is False:
        report_type = type_judge(lstKwds_need_judge=report_overview,
                                 conf_path='conf')
        class_report.repo_type = type_judge(lstKwds_need_judge=report_overview,
                                            conf_path='conf')  # class
    if report_type_lock is True:
        report_type = '肺功能'
        class_report.repo_type = '肺功能'  # class
    path_suffix = f'-{report_type}'

    conf_path = f'conf/{path_prefix}{path_suffix}.conf'
    if os.path.exists(conf_path) is False:
        # print('配置文件不存在')
        return '错误：目前暂不支持此医院的此种报告'
    img_feature_path = f'OCR_IMG/Feature_IMG/{path_prefix}{path_suffix}.jpg'
    if os.path.exists(img_feature_path) is False:
        # print('特征图片不存在')
        return '错误：缺少特征图片，无法匹配，请等候开发者后续维护'

    # 读取配置
    # conf_path = 'conf/bj-aerospace-blood-normal.conf'
    conf = configparser_custom()
    conf.read(conf_path, 'UTF-8')
    boxes_conf = conf.items("boxes")
    name_list = []
    # box_list[] 格式:[左，右，上，下]
    box_list = []
    for w in range(len(boxes_conf)):
        name_list.append(boxes_conf[w][0])
        box_list.append(boxes_conf[w][1].split(','))
        box_list[w] = list(map(int, box_list[w]))

    # 新，读配置文件为dict，方便使用配置文件的标题对数据组合进行自动标识
    dict_boxes_conf = conf.as_dict()["boxes"]
    '''
    for list_i in dict_boxes_conf.items():
        print(list_i)
    '''
    list_title = []
    for list_i in dict_boxes_conf:
        remove_digits = str.maketrans('', '', digits)
        list_i = list_i.translate(remove_digits)
        list_title.append(list_i)

    # 特征匹配准备裁剪

    # img_template = cv2.imread(img_feature_path, 0)
    img_template = PRE_pross.cv_imread_chs(img_feature_path)

    # 灰度化
    img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    img_need_pross = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

    img_small_1k, ratio = PRE_pross.zoom_to_1k(img_need_pross)  # 屏幕匹配提速

    cv2.imwrite('temp/DEMO/1k.jpg', img_small_1k)
    # [旧]correct_points, knn_result = knn_match_old(img_template, img_small_1k, demo)
    correct_matrix, knn_result = PRE_pross.knn_match_new(template_img=img_template,
                                                         img_need_match=img_small_1k,
                                                         demo=demo_or_not)
    if knn_result == 2874734:
        return '错误：未能成功探测布局，建议重新拍摄'
    # 以下是新变换办法，直接用单应性矩阵变换后直接裁剪得到目标图像（能解决老办法不能自动旋转的问题）
    correct_matrix[0][2] = correct_matrix[0][2] / ratio
    correct_matrix[1][2] = correct_matrix[1][2] / ratio
    correct_matrix[2][0] = correct_matrix[2][0] * ratio
    correct_matrix[2][1] = correct_matrix[2][1] * ratio
    img_screen_cut = cv2.warpPerspective(img_need_pross, correct_matrix, (round((img_template.shape[1]) / ratio),
                                                                          round((img_template.shape[0]) / ratio)))
    # cv2.imwrite('res.jpg', img_screen_cut)
    # plt.imshow(img_screen_cut, 'gray'), plt.show()
    img_screen_cut_1k, ratio_outdate = PRE_pross.zoom_to_1k(img_screen_cut)

    cv2.imwrite('temp/region.jpg', img_screen_cut_1k)  # 适配开源血常规

    # 将用户信息识别滞后，提升识别概率

    class_img_screen_cut_1k = Img(IP_Address=ocr_ip, img_f=img_screen_cut_1k)

    usr_info_response = class_img_screen_cut_1k.Send_To_OCR()

    if usr_info_response is 'OCROFFLINE':
        return '错误：OCR离线'
    elif len(usr_info_response.json()["results"]) == 0:
        # print('OCRERR')
        return f'错误：OCR没有正常工作,\n{usr_info_response.text}'
    # pprint(usr_info_response.json()["results"][0]["data"])
    usr_info_overview = []
    for i in range(len(usr_info_response.json()["results"][0]["data"])):
        usr_info_overview.append(usr_info_response.json()["results"][0]["data"][i]['text'])
    patient_name = PRE_pross.charactor_match_count_name_age(usr_info_overview, '名：')
    class_report.name = PRE_pross.charactor_match_count_name_age(usr_info_overview, '名：')  # class
    if patient_name:
        # 以防万一把所有冒号前的东西重新统一
        patient_name = re.sub(r'.*：', '姓名：', patient_name)
        class_report.name = re.sub(r'.*：', '姓名：', patient_name)  # class
    else:
        patient_name = '姓名：'
        class_report.name = '姓名：'  # class
    patient_sex = PRE_pross.charactor_match_count_sex(usr_info_overview, '别：')
    class_report.sex = PRE_pross.charactor_match_count_sex(usr_info_overview, '别：')  # class
    if patient_sex:
        patient_sex = re.sub(r'.*：', '性别：', patient_sex)
        class_report.sex = re.sub(r'.*：', '性别：', patient_sex)  # class
    else:
        patient_name = '性别：'
        class_report.sex = '性别：'  # class
    patient_age = PRE_pross.charactor_match_count_name_age(usr_info_overview, '龄：')
    class_report.age = PRE_pross.charactor_match_count_name_age(usr_info_overview, '龄：')  # class
    if patient_age:
        patient_age = re.sub(r'.*：', '年龄：', patient_age)
        class_report.age = re.sub(r'.*：', '年龄：', patient_age)  # class
    else:
        patient_name = '年龄：'
        class_report.age = '年龄：'  # class

    repo_date = PRE_pross.charactor_match_count_name_age(usr_info_overview, '报告日期：')
    class_report.repo_data = PRE_pross.charactor_match_count_name_age(usr_info_overview, '报告日期：')  # class
    if repo_date:
        repo_date = re.sub(r'.*：', '报告日期：', repo_date)
        class_report.repo_data = repo_date = re.sub(r'.*：', '报告日期：', repo_date)  # class
    else:
        repo_date = PRE_pross.charactor_match_count_name_age(usr_info_overview, '报告时间：')
        class_report.repo_data = PRE_pross.charactor_match_count_name_age(usr_info_overview, '报告时间：')  # class
        if repo_date:
            repo_date = re.sub(r'.*：', '报告时间：', repo_date)
            class_report.repo_data = re.sub(r'.*：', '报告时间：', repo_date)  # class
        else:
            repo_date = '报告日期：'
            class_report.repo_data = '报告日期：'  # class

            # 用户信息识别结束

    img_screen_cut = PRE_pross.length_width_ratio_correct(img_template=img_template,
                                                          img_input=img_screen_cut)  # 长宽比校正

    # 下面开始按照比例裁剪识别区域
    img_element = PRE_pross.mask_processing_new(img_input=img_screen_cut,
                                                boxes_coordinate_xy=box_list,
                                                demo_or_not=demo_or_not,
                                                type_char='repo',
                                                out_name='report')  # 根据配置裁剪数据区
    # 拓宽防止ocr不识别的bug
    for i in range(len(img_element)):
        img_element[i] = PRE_pross.image_border(img_input=img_element[i],
                                                dst='0')
    # 取得识别结果

    answer = []
    '''
    # 非多线程

    for n in range(len(img_element)):
        response = net_OCR(img_element[n], ocr_ip)
        if response is 'OCROFFLINE':
            return '错误：OCR离线'
        elif len(response.json()["results"]) == 0:
            # print('OCRERR')
            return f'错误：OCR未正常工作,{response.text}'

        # pprint(response.json()["results"][0]["data"])
        answer.append(response)
        answer[n] = answer[n].json()["results"]
    '''
    # 多线程
    ocr_pool = ThreadPoolExecutor(max_workers=10)
    answer_muity = [0] * (len(img_element))
    jobs = [0] * (len(img_element))
    for n in range(len(img_element)):
        rec_element = Img(IP_Address=ocr_ip, img_f=img_element[n])
        jobs[n] = ocr_pool.submit(rec_element.Send_To_OCR)
    ocr_pool.shutdown(wait=True)
    # 多线程结束

    for n in range(len(img_element)):
        # answer_muity[n] = get_result(jobs[n])
        answer_muity[n] = jobs[n].result()
        if answer_muity[n] is 'OCROFFLINE':
            return '错误：OCR离线'
        elif len(answer_muity[n].json()["results"]) == 0:
            # print('OCRERR')
            return f'错误：OCR未正常工作,{answer_muity[n].text}'

        # pprint(response.json()["results"][0]["data"])
        answer.append(answer_muity[n])
        answer[n] = answer[n].json()["results"]

    # 判断是否识别完全

    # print(answer[1][0]['data'][1]['text'])
    # 以下开始处理识别回传数据
    # 格式说明： ?_position[],先长后高，左上开始顺时针4点

    # 开始根据坐标对齐

    answer_need_to_alligin = AnswerToReport(ocr_answer=answer, list_title=list_title)
    class_report.list = answer_need_to_alligin.to_list()

    # 加入附加信息
    '''
    bloodtest_single = OrderedDict()
    bloodtest_single["name"] = f'{hospital}{path_suffix}'
    bloodtest_single["value"] = patient_name
    bloodtest_single["range"] = patient_sex
    bloodtest_single["alias"] = '空白信息2'
    bloodtest_single["unit"] = patient_age
    bloodtest_list.append(bloodtest_single)
    '''

    test_dict_class = {
        'hospital': f'{class_report.hos_name}',
        'repo_type': f'{class_report.repo_type}',
        'repo_date': f'{class_report.repo_data}',
        'name': f'{class_report.name}',
        'age': f'{class_report.age}',
        'sex': f'{class_report.sex}',
        'bloodtest': class_report.list,
        'write_time': f'{datetime.now()}',
        'explain': {
            'used': True,
            'details': "json生成测试",
        }
    }
    # json_str = json.dumps(test_dict, ensure_ascii=False, indent=4)
    json_str = json.dumps(test_dict_class, ensure_ascii=False, indent=4)  # class
    with open('test_data.json', 'w') as json_file:
        json_file.write(json_str)
    return json_str


if __name__ == '__main__':
    # 清空临时文件
    shutil.rmtree('temp')
    os.mkdir('temp')
    os.mkdir('temp/DEMO')
    os.mkdir('temp/ocr_result')
    os.mkdir('temp/DEMO/mask')

    img_orig_path = 'OCR_IMG/Input_IMG/zs-blood-normal.jpg'
    img_input = PRE_pross.cv_imread_chs(img_orig_path)

    main_pross(cvimg=img_input,
               demo_or_not=1,
               hospital_lock=False,
               report_type_lock=False)
